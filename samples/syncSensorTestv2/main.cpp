/*
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ArgusHelpers.h"
#include "Error.h"
#include <stdio.h>
#include <stdlib.h>
#include <Argus/Argus.h>
#include <Argus/Ext/BayerAverageMap.h>
#include <Argus/Ext/NonLinearHistogram.h>
#include <EGLStream/EGLStream.h>
#include "PreviewConsumer.h"
#include "CommonOptions.h"
#include "CaptureConsumer.h"
#include <algorithm>
#define FILE_DIR "/home/asus/Desktop/result/"
#define NENOTOMILLI 1000000.0f
#define THRESHOLD 5000
// Debug print macros.
#define PRODUCER_PRINT(...) printf("PRODUCER: " __VA_ARGS__)
#define CONSUMER_PRINT(...) printf("CONSUMER: " __VA_ARGS__)


#define EXIT_IF_TRUE(val,msg)   \
        {if ((val)) {printf("%s\n",msg); return false;}}
#define EXIT_IF_NULL(val,msg)   \
        {if (!val) {printf("%s\n",msg); return false;}}
#define EXIT_IF_NOT_OK(val,msg) \
        {if (val!=Argus::STATUS_OK) {printf("%s\n",msg); return false;}}

enum maxCamDevice
{
    LEFT_CAM_DEVICE  = 0,
    RIGHT_CAM_DEVICE = 1,
    MAX_CAM_DEVICE = 2
};
using namespace Argus;

namespace ArgusSamples
{

// Globals.
EGLDisplayHolder g_display;

// Constants
const float BAYER_CLIP_COUNT_MAX = 0.25f;
const float BAYER_MISSING_SAMPLE_TOLERANCE = 0.10f;
const float BAYER_HISTOGRAM_CLIP_PERCENTILE = 0.10f;

/*******************************************************************************
 * Extended options class to add additional options specific to this sample.
 ******************************************************************************/
class UserAutoWhiteBalanceSampleOptions : public CommonOptions
{
public:
    UserAutoWhiteBalanceSampleOptions(const char *programName)
        : CommonOptions(programName,
                        ArgusSamples::CommonOptions::Option_D_CameraDevice |
                        ArgusSamples::CommonOptions::Option_M_SensorMode |
                        ArgusSamples::CommonOptions::Option_R_WindowRect |
                        ArgusSamples::CommonOptions::Option_F_FrameCount)
        , m_useAverageMap(false)
    {
        addOption(createValueOption
            ("useaveragemap", 'a', "0 or 1", "Use Average Map (instead of Bayer Histogram).",
             m_useAverageMap));
    }

    bool useAverageMap() const { return m_useAverageMap.get(); }

protected:
    Value<bool> m_useAverageMap;
};

/**
 * RAII class for app teardown
 */
class UserAutoWhiteBalanceTeardown
{
public:
    CameraProvider* m_cameraProvider;
    PreviewConsumerThread* m_previewConsumerThread;
    OutputStream* m_stream;

    UserAutoWhiteBalanceTeardown()
    {
        m_cameraProvider = NULL;
        m_previewConsumerThread = NULL;
        m_stream = NULL;
    }

    ~UserAutoWhiteBalanceTeardown()
    {
        shutdown();
    }

private:
    void shutdown()
    {
        // Destroy the output streams (stops consumer threads).
        if (m_stream != NULL)
            m_stream->destroy();

        // Wait for the consumer threads to complete.
        if (m_previewConsumerThread != NULL) {
            PROPAGATE_ERROR_CONTINUE(m_previewConsumerThread->shutdown());
            delete m_previewConsumerThread;
            m_previewConsumerThread = NULL;
        }

        // Shut down Argus.
        if (m_cameraProvider != NULL)
            m_cameraProvider->destroy();

        // Shut down the window (destroys window's EGLSurface).
        Window::getInstance().shutdown();

        // Cleanup the EGL display
        PROPAGATE_ERROR_CONTINUE(g_display.cleanup());
    }
};

/*
 * Program: userAutoWhiteBalance
 * Function: To display 101 preview images to the device display to illustrate a
 *           Grey World White Balance technique for adjusting the White Balance in real time
 *           through the use of setWbGains() and setAwbMode(AWB_MODE_MANUAL) calls.
 */

static bool execute(const UserAutoWhiteBalanceSampleOptions& options)
{
    // Initialize the window and EGL display.
    Window::getInstance().setWindowRect(options.windowRect());
    PROPAGATE_ERROR(g_display.initialize(Window::getInstance().getEGLNativeDisplay()));

    /*
     * Set up Argus API Framework, identify available camera devices, create
     * a capture session for the first available device, and set up the event
     * queue for completed events
     */

    CameraProvider* m_cameraProvider = CameraProvider::create();
    ICameraProvider *iCameraProvider = interface_cast<ICameraProvider>(m_cameraProvider);
    EXIT_IF_NULL(iCameraProvider, "Cannot get core camera provider interface");
    printf("1Argus Version: %s\n", iCameraProvider->getVersion().c_str());

    // Get the selected camera device and sensor mode.
    std::vector<CameraDevice*> cameraDevices;
    iCameraProvider->getCameraDevices(&cameraDevices);
    if (cameraDevices.size() < 2)
        ORIGINATE_ERROR("Must have at least 2 sensors available");

    ICameraProperties *iCameraProperties = interface_cast<ICameraProperties>(cameraDevices[0]);
    if (!iCameraProperties)
        ORIGINATE_ERROR("Failed to get ICameraProperties interface");

    ISensorMode *iSensorMode;
    std::vector<SensorMode*> sensorModes;
    iCameraProperties->getBasicSensorModes(&sensorModes);
    if (sensorModes.size() == 0)
        ORIGINATE_ERROR("Failed to get sensor modes");

    PRODUCER_PRINT("Available Sensor modes :\n");
    for (uint32_t i = 0; i < sensorModes.size(); i++) {
        iSensorMode = interface_cast<ISensorMode>(sensorModes[i]);
        Size2D<uint32_t> resolution = iSensorMode->getResolution();
        PRODUCER_PRINT("[%u] W=%u H=%u\n", i, resolution.width(), resolution.height());
    }
    
    // Create the CaptureSession using the selected device.
    std::vector<EventType> eventTypes;
    eventTypes.push_back(EVENT_TYPE_CAPTURE_COMPLETE);
    eventTypes.push_back(EVENT_TYPE_ERROR);
    /* Seems there is bug in Argus, which drops EVENT_TYPE_ERROR if all
    3 events are not set. Set it for now */
    eventTypes.push_back(EVENT_TYPE_CAPTURE_STARTED);
    
    IEventQueue *iQueue[MAX_CAM_DEVICE];
    ICaptureSession* iCaptureSession[MAX_CAM_DEVICE];
    CaptureSession* captureSession[MAX_CAM_DEVICE];
    IEventProvider *iEventProvider[MAX_CAM_DEVICE];
    EventQueue *queue[MAX_CAM_DEVICE];
    
    for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++) {
        captureSession[i] = iCameraProvider->createCaptureSession(cameraDevices[i]);
        iCaptureSession[i] = interface_cast<ICaptureSession>(captureSession[i]);
        EXIT_IF_NULL(iCaptureSession[i], "Cannot get Capture Session Interface");
        
        iEventProvider[i] = interface_cast<IEventProvider>(captureSession[i]);
        EXIT_IF_NULL(iEventProvider[i], "iEventProvider is NULL");
        
        queue[i] = iEventProvider[i]->createEventQueue(eventTypes);
        iQueue[i] = interface_cast<IEventQueue>(queue[i]);
        EXIT_IF_NULL(iQueue[i], "event queue interface is NULL");
   }
    
    /*
     * Creates the stream between the Argus camera image capturing
     * sub-system (producer) and the image acquisition code (consumer)
     * preview thread.  A consumer object is created from the stream
     * to be used to request the image frame.  A successfully submitted
     * capture request activates the stream's functionality to eventually
     * make a frame available for acquisition, in the preview thread,
     * and then display it on the device screen.
     */

    OutputStream *stream[MAX_CAM_DEVICE];
    IEGLOutputStream *iEGLOutputStream[MAX_CAM_DEVICE];
    PreviewConsumerThread* m_previewConsumerThread[MAX_CAM_DEVICE];
    IRequest *iRequest[MAX_CAM_DEVICE];
    Request *request[MAX_CAM_DEVICE];
    
    IRequest *ijRequest[MAX_CAM_DEVICE];
    Request *jrequest[MAX_CAM_DEVICE];
    OutputStream *capture_stream[MAX_CAM_DEVICE];

    std::vector<EGLStreamKHR> eglStreams;
    EGLStream::IFrameConsumer *iFrameConsumer[MAX_CAM_DEVICE];
    EGLStream::FrameConsumer *consumer[MAX_CAM_DEVICE];
    for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++) {
        UniqueObj<OutputStreamSettings> streamSettings(
        iCaptureSession[i]->createOutputStreamSettings(STREAM_TYPE_EGL));
        
        IEGLOutputStreamSettings *iEGLStreamSettings =
            interface_cast<IEGLOutputStreamSettings>(streamSettings);
        EXIT_IF_NULL(iEGLStreamSettings, "Cannot get IEGLOutputStreamSettings Interface");
        
        iEGLStreamSettings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
        iEGLStreamSettings->setResolution(Size2D<uint32_t>(options.windowRect().width(),
                                                       options.windowRect().height()));
        iEGLStreamSettings->setEGLDisplay(g_display.get());    
        iEGLStreamSettings->setMetadataEnable(true);
        stream[i] = iCaptureSession[i]->createOutputStream(streamSettings.get());
        iEGLOutputStream[i] = interface_cast<IEGLOutputStream>(stream[i]);
        EXIT_IF_NULL(iEGLOutputStream[i], "Cannot get IEGLOutputStream Interface");
        eglStreams.push_back(iEGLOutputStream[i]->getEGLStream());

        capture_stream[i] = iCaptureSession[i]->createOutputStream(streamSettings.get());
        EXIT_IF_NULL(stream, "Failed to create EGLOutputStream");
      /*  
        consumer[i] = EGLStream::FrameConsumer::create(capture_stream[i]);

        iFrameConsumer[i] = Argus::interface_cast<EGLStream::IFrameConsumer>(consumer[i]);
        EXIT_IF_NULL(iFrameConsumer[i], "Failed to initialize Consumer");
    */

        request[i] = iCaptureSession[i]->createRequest(CAPTURE_INTENT_STILL_CAPTURE);
        iRequest[i] = interface_cast<IRequest>(request[i]);
        EXIT_IF_NULL(iRequest[i], "Failed to get capture request interface");

        jrequest[i] = iCaptureSession[i]->createRequest(CAPTURE_INTENT_STILL_CAPTURE);
        ijRequest[i] = interface_cast<IRequest>(jrequest[i]);
        EXIT_IF_NULL(iRequest[i], "Failed to get capture request interface");
    }
    
    m_previewConsumerThread[0] = new PreviewConsumerThread(g_display.get(), eglStreams,
                                     PreviewConsumerThread::LAYOUT_SPLIT_VERTICAL,
                                     false );
    PROPAGATE_ERROR(m_previewConsumerThread[0]->initialize());
    PROPAGATE_ERROR(m_previewConsumerThread[0]->waitRunning());

    CaptureConsumerThread* jpegConsumer[MAX_CAM_DEVICE];
    for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++) {
        jpegConsumer[i] = new CaptureConsumerThread(capture_stream[i],i);
        PROPAGATE_ERROR(jpegConsumer[i]->initialize());
        PROPAGATE_ERROR(jpegConsumer[i]->waitRunning());
    }
    // Set the sensor mode in the request.
    static uint32_t SENSOR_MODE   = 4;
    static int      CAPTURE_FPS   = 30;
    /* Check and set sensor mode */

    if (SENSOR_MODE >= sensorModes.size())
        ORIGINATE_ERROR("Sensor mode index is out of range");
    SensorMode *sensorMode = sensorModes[SENSOR_MODE];
    iSensorMode = interface_cast<ISensorMode>(sensorMode);
    for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++) {
        ISourceSettings *iSourceSettings =
            interface_cast<ISourceSettings>(iRequest[i]->getSourceSettings());
        iSourceSettings->setSensorMode(sensorMode);
        iSourceSettings->setFrameDurationRange(Range<uint64_t>(1e9/CAPTURE_FPS));
        EXIT_IF_NOT_OK(iRequest[i]->enableOutputStream(stream[i]),"Failed to enable stream in capture request");
        EXIT_IF_NOT_OK(iCaptureSession[i]->repeat(request[i]), "Unable to submit repeat() request");
    }
    for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++) {
        ISourceSettings *iSourceSettings =
            interface_cast<ISourceSettings>(ijRequest[i]->getSourceSettings());
        iSourceSettings->setSensorMode(sensorMode);
        iSourceSettings->setFrameDurationRange(Range<uint64_t>(1e9/CAPTURE_FPS));
        EXIT_IF_NOT_OK(ijRequest[i]->enableOutputStream(capture_stream[i]),"Failed to enable stream in capture request");
    }
    /*
     * Using the image capture event metadata, acquire the bayer histogram and then compute
     * a weighted average for each channel.  Use these weighted averages to create a White
     * Balance Channel Gain array to use for setting the manual white balance of the next
     * capture request.
     */
    printf("4Argus Version: %s options.frameCount() %d \n", iCameraProvider->getVersion().c_str(), options.frameCount());

    uint32_t frameCaptureLoop = 0;
    bool isAdjust = false;
    uint64_t orinframeduration = 1e9/CAPTURE_FPS;
    int adjustCamIndex = 0;
    unsigned long long sensorTime[2];
    uint32_t captureId[2];
    uint64_t frameduration[2];
    long long diff;
    char filepath[200];
    int count = 0;
    int onetime = 1;
    int countsForCaptureLoop = 120;
    Argus::Status status;
    while (frameCaptureLoop < countsForCaptureLoop)
    {
        // Keep PREVIEW display window serviced
        Window::getInstance().pollEvents();

        const uint64_t FIVE_SECONDS = 5000000000;
        iEventProvider[0]->waitForEvents(queue[0], FIVE_SECONDS);
        EXIT_IF_TRUE(iQueue[0]->getSize() == 0, "No events in queue");

        const Event* event = iQueue[0]->getEvent(iQueue[0]->getSize() - 1);
        const IEvent* iEvent = interface_cast<const IEvent>(event);
        iEventProvider[1]->waitForEvents(queue[1], FIVE_SECONDS);
        EXIT_IF_TRUE(iQueue[1]->getSize() == 0, "No events in queue");
        const Event* event1 = iQueue[1]->getEvent(iQueue[1]->getSize() - 1);
        const IEvent* iEvent1 = interface_cast<const IEvent>(event1);
        if (!iEvent)
            printf("Error : Failed to get IEvent interface\n");
        else {
            if (iEvent->getEventType() == EVENT_TYPE_CAPTURE_COMPLETE) {
                frameCaptureLoop++;
                const IEventCaptureComplete* iEventCaptureComplete =
                    interface_cast<const IEventCaptureComplete>(event);
                EXIT_IF_NULL(iEventCaptureComplete, "Failed to get EventCaptureComplete Interface");

                const CaptureMetadata* metaData = iEventCaptureComplete->getMetadata();
                const ICaptureMetadata* iMetadata = interface_cast<const ICaptureMetadata>(metaData);

                EXIT_IF_NULL(iMetadata, "Failed to get CaptureMetadata Interface");
                printf("cam 0EVENT_TYPE_CAPTURE_COMPLETE\n");
                //const TimeValue sensorTime = TimeValue::fromNSec(iMetadata->getSensorTimestamp());
                
                sensorTime[0] = iMetadata->getSensorTimestamp();
                frameduration[0] = iMetadata->getFrameDuration();
                captureId[0] = iEvent->getCaptureId();
                
                if (iEvent1->getEventType() == EVENT_TYPE_CAPTURE_COMPLETE) {
                    const IEventCaptureComplete* iEventCaptureComplete1 =
                       interface_cast<const IEventCaptureComplete>(event1);
                    EXIT_IF_NULL(iEventCaptureComplete, "Failed to get EventCaptureComplete Interface");

                    const CaptureMetadata* metaData1 = iEventCaptureComplete1->getMetadata();
                    const ICaptureMetadata* iMetadata1 = interface_cast<const ICaptureMetadata>(metaData1);

                    EXIT_IF_NULL(iMetadata1, "Failed to get CaptureMetadata Interface");
                    printf("cam 1 EVENT_TYPE_CAPTURE_COMPLETE\n");
                    //const TimeValue sensorTime = TimeValue::fromNSec(iMetadata->getSensorTimestamp());
                    
                    sensorTime[1] = iMetadata1->getSensorTimestamp();
                    frameduration[1] = iMetadata1->getFrameDuration();
                    captureId[1] = iEvent1->getCaptureId();
                    
                    diff = labs(sensorTime[1] - sensorTime[0]);
                    printf("sensorTime %lld - %lld = %lld (%f)\n",
                        sensorTime[0],
                        sensorTime[1],
                        diff,
                        sensorTime[1]/1000000.0f - sensorTime[0]/1000000.0f);
                   //printf("cam2 id %d timestamps %.2f ms\n",iEvent1->getCaptureId() ,sensorTime1/1000000.0f);
                    printf("duration %ld isAdjust %d captrueId[%d, %d] timestamps [%.2f, %.2f]ms diff %llu frameduration [%ld, %ld]\n",
                        orinframeduration,
                        isAdjust,
                        captureId[0],
                        captureId[1],
                        sensorTime[0]/NENOTOMILLI,
                        sensorTime[1]/NENOTOMILLI,
                        diff,
                        frameduration[0],
                        frameduration[1]);
                        /*for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++) {
                            Argus::UniqueObj<EGLStream::Frame> frame(iFrameConsumer[i]->acquireFrame(FIVE_SECONDS, &status));

                            EGLStream::IFrame *iFrame = Argus::interface_cast<EGLStream::IFrame>(frame);
                            EXIT_IF_NULL(iFrame, "Failed to get IFrame interface");

                            EGLStream::Image *image = iFrame->getImage();
                            EXIT_IF_NULL(image, "Failed to get Image from iFrame->getImage()");
                            EGLStream::IImageJPEG *iImageJPEG = Argus::interface_cast<EGLStream::IImageJPEG>(image);
	                    if(!iImageJPEG)
	                    {
	                            ORIGINATE_ERROR("Failed to get ImageJPEG Interface");
                            }

	                        sprintf(filepath,FILE_DIR "[Cam%d][CaptureId%d][Duration_%ld][Timestamp_%.2fms][diff_%.2fms].jpg",
	                            i,
	                            captureId[i],
	                            frameduration[i],
	                            sensorTime[i]/NENOTOMILLI,
	                            diff/NENOTOMILLI);
	                         
                            Argus::Status status = iImageJPEG->writeJPEG(filepath);
                            if(status != Argus::STATUS_OK)
                            {
                                ORIGINATE_ERROR("Failed to write JPEG");
                            }
                            //iFrameLeft = interface_cast<IFrame>(frameleft);
                            iFrame->releaseFrame();
                          }*/
                    if (isAdjust)
                    {
                        ISourceSettings *iSourceSettings =
                            interface_cast<ISourceSettings>(iRequest[adjustCamIndex]->getSourceSettings());
                        iSourceSettings->setFrameDurationRange(Range<uint64_t>(1e9/CAPTURE_FPS));
                        EXIT_IF_NOT_OK(iCaptureSession[adjustCamIndex]->repeat(request[adjustCamIndex]), "Unable to submit repeat() request");   
                        //#define NENOTOMILLI 1000000.0f
                        //#define threshold 5000
                        if(labs(orinframeduration - frameduration[adjustCamIndex]) < THRESHOLD)
                        {
                            isAdjust = false;
                            printf("-----------adjust timestamp back done-------------------------\n");
                        }
                        printf("adjust timestamp back\n");
                    }
                    if (diff > 10*NENOTOMILLI && !isAdjust)
                    {
                        adjustCamIndex = sensorTime[1] > sensorTime[0] ? 0 : 1;
                        ISourceSettings *iSourceSettings =
                            interface_cast<ISourceSettings>(iRequest[adjustCamIndex]->getSourceSettings());
                        iSourceSettings->setFrameDurationRange(Range<uint64_t>(1e9/CAPTURE_FPS + diff/2));
                        EXIT_IF_NOT_OK(iCaptureSession[adjustCamIndex]->repeat(request[adjustCamIndex]), "Unable to submit repeat() request");
                        isAdjust = true;
                        printf("adjust timestamp of Cam %d captrueId[%d, %d] timestamps [%.2f, %.2f]ms diff %.2f\n",
                            adjustCamIndex,
                            captureId[0],
                            captureId[1],
                            sensorTime[0]/NENOTOMILLI,
                            sensorTime[1]/NENOTOMILLI,
                            diff/NENOTOMILLI);
                    }
                    else if ( diff < 10*NENOTOMILLI && !isAdjust && onetime && captureId[0] > 100)
                    {
                        onetime = 0;
                        printf("dump image of isAdjust %d captrueId[%d, %d] timestamps [%.2f, %.2f]ms diff %.2f\n",
                            isAdjust,
                            captureId[0],
                            captureId[1],
                            sensorTime[0]/NENOTOMILLI,
                            sensorTime[1]/NENOTOMILLI,
                            diff/NENOTOMILLI);
		            //iCaptureSession[0]->capture(jrequest[0]);
                            //iCaptureSession[1]->capture(jrequest[1]);
		            break;
                      }
               }
            } else if (iEvent->getEventType() == EVENT_TYPE_CAPTURE_STARTED) {
                /* ToDo: Remove the empty after the bug is fixed */
                continue;
            } else if (iEvent->getEventType() == EVENT_TYPE_ERROR) {
                const IEventError* iEventError =
                    interface_cast<const IEventError>(event);
                EXIT_IF_NOT_OK(iEventError->getStatus(), "ERROR event");
            } else {
                printf("WARNING: Unknown event. Continue\n");
            }
        }
    }
    iCaptureSession[0]->capture(jrequest[0]);
    iCaptureSession[1]->capture(jrequest[1]);
    sleep(3);
    //iSession->stopRepeat is cleaned with captureSession RAII
    for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++) {
        iCaptureSession[i]->stopRepeat();
        iCaptureSession[i]->waitForIdle();
        if(capture_stream[i] != NULL)
            capture_stream[i]->destroy();
        if(stream[i] != NULL)
            stream[i]->destroy();
        if (jpegConsumer[i] != NULL) {
            PROPAGATE_ERROR_CONTINUE(jpegConsumer[i]->shutdown());
            delete jpegConsumer[i];
            jpegConsumer[i] = NULL;
        }
    }
    if (m_previewConsumerThread[0] != NULL) {
            PROPAGATE_ERROR_CONTINUE(m_previewConsumerThread[0]->shutdown());
            delete m_previewConsumerThread[0];
            m_previewConsumerThread[0] = NULL;
    }
    if (m_cameraProvider != NULL)
        m_cameraProvider->destroy();
    Window::getInstance().shutdown();
    PROPAGATE_ERROR_CONTINUE(g_display.cleanup());
    return true;
}

}; // namespace ArgusSamples

int main(int argc, char **argv)
{
    ArgusSamples::UserAutoWhiteBalanceSampleOptions options(basename(argv[0]));
    if (!options.parse(argc, argv))
        return EXIT_FAILURE;
    if (options.requestedExit())
        return EXIT_SUCCESS;

    if (!ArgusSamples::execute(options))
    {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
