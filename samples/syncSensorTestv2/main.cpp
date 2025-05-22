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
#include <algorithm>
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
    UserAutoWhiteBalanceTeardown appTearDown;
    BayerTuple<float> precedingAverages(1.0f, 1.5f, 1.5f, 1.5f);

    // Initialize the window and EGL display.
    Window::getInstance().setWindowRect(options.windowRect());
    PROPAGATE_ERROR(g_display.initialize(Window::getInstance().getEGLNativeDisplay()));

    /*
     * Set up Argus API Framework, identify available camera devices, create
     * a capture session for the first available device, and set up the event
     * queue for completed events
     */

    appTearDown.m_cameraProvider = CameraProvider::create();
    ICameraProvider *iCameraProvider = interface_cast<ICameraProvider>(appTearDown.m_cameraProvider);
    EXIT_IF_NULL(iCameraProvider, "Cannot get core camera provider interface");
    printf("1Argus Version: %s\n", iCameraProvider->getVersion().c_str());

    // Get the selected camera device and sensor mode.
    /*CameraDevice* cameraDevice = ArgusHelpers::getCameraDevice(appTearDown.m_cameraProvider, options.cameraDeviceIndex());

    if (!cameraDevice)
        ORIGINATE_ERROR("Selected camera device is not available");
    */
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
    /*
    SensorMode* sensorMode = ArgusHelpers::getSensorMode(cameraDevice[0], options.sensorModeIndex());
    ISensorMode *iSensorMode = interface_cast<ISensorMode>(sensorMode);
    if (!iSensorMode)
        ORIGINATE_ERROR("Selected sensor mode not available");
    */
    
    // Create the CaptureSession using the selected device.
/*    UniqueObj<CaptureSession> captureSession(iCameraProvider->createCaptureSession(cameraDevices[1]));
    ICaptureSession* iSession = interface_cast<ICaptureSession>(captureSession);
    EXIT_IF_NULL(iSession, "Cannot get Capture Session Interface");

    IEventProvider *iEventProvider = interface_cast<IEventProvider>(captureSession);
    EXIT_IF_NULL(iEventProvider, "iEventProvider is NULL");
    
*/
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
    OutputStreamSettings* streamSettings[MAX_CAM_DEVICE];
    for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++) {
        captureSession[i] = iCameraProvider->createCaptureSession(cameraDevices[i]);
        iCaptureSession[i] = interface_cast<ICaptureSession>(captureSession[i]);
        EXIT_IF_NULL(iCaptureSession[i], "Cannot get Capture Session Interface");
        iEventProvider[i] = interface_cast<IEventProvider>(captureSession[i]);
        EXIT_IF_NULL(iEventProvider[i], "iEventProvider is NULL");
        queue[i] = iEventProvider[i]->createEventQueue(eventTypes);
        iQueue[i] = interface_cast<IEventQueue>(queue[i]);
        EXIT_IF_NULL(iQueue[i], "event queue interface is NULL");

       streamSettings[i] = iCaptureSession[i]->createOutputStreamSettings(STREAM_TYPE_EGL);
       IEGLOutputStreamSettings *iEGLStreamSettings =
          interface_cast<IEGLOutputStreamSettings>(streamSettings[i]);
       EXIT_IF_NULL(iEGLStreamSettings, "Cannot get IEGLOutputStreamSettings Interface");
       iEGLStreamSettings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
       iEGLStreamSettings->setResolution(Size2D<uint32_t>(options.windowRect().width(),
                                                          options.windowRect().height()));
       iEGLStreamSettings->setEGLDisplay(g_display.get());
   }
    
/*    
    UniqueObj<EventQueue> queue(iEventProvider->createEventQueue(eventTypes));
    IEventQueue *iQueue = interface_cast<IEventQueue>(queue);
    EXIT_IF_NULL(iQueue, "event queue interface is NULL");
*/
    /*
     * Creates the stream between the Argus camera image capturing
     * sub-system (producer) and the image acquisition code (consumer)
     * preview thread.  A consumer object is created from the stream
     * to be used to request the image frame.  A successfully submitted
     * capture request activates the stream's functionality to eventually
     * make a frame available for acquisition, in the preview thread,
     * and then display it on the device screen.
     */
    printf("2Argus Version: %s\n", iCameraProvider->getVersion().c_str());

    /*
    UniqueObj<OutputStreamSettings> streamSettings(
        iSession->createOutputStreamSettings(STREAM_TYPE_EGL));

    IEGLOutputStreamSettings *iEGLStreamSettings =
        interface_cast<IEGLOutputStreamSettings>(streamSettings);
    EXIT_IF_NULL(iEGLStreamSettings, "Cannot get IEGLOutputStreamSettings Interface");

    iEGLStreamSettings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
    iEGLStreamSettings->setResolution(Size2D<uint32_t>(options.windowRect().width(),
                                                       options.windowRect().height()));
    iEGLStreamSettings->setEGLDisplay(g_display.get());
    */

    OutputStream *stream[MAX_CAM_DEVICE];
    IEGLOutputStream *iEGLOutputStream[MAX_CAM_DEVICE];
    PreviewConsumerThread* m_previewConsumerThread[MAX_CAM_DEVICE];
    IRequest *iRequest[MAX_CAM_DEVICE];
    //UniqueObj<Request> *request[MAX_CAM_DEVICE];
    Request *request[MAX_CAM_DEVICE];
    std::vector<EGLStreamKHR> eglStreams;
    for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++) {
        printf("3Argus Version: %s\n", iCameraProvider->getVersion().c_str());
        UniqueObj<OutputStreamSettings> streamSettings(
        iCaptureSession[i]->createOutputStreamSettings(STREAM_TYPE_EGL));
    printf("3Argus Version: %s\n", iCameraProvider->getVersion().c_str());
    IEGLOutputStreamSettings *iEGLStreamSettings =
        interface_cast<IEGLOutputStreamSettings>(streamSettings);
    EXIT_IF_NULL(iEGLStreamSettings, "Cannot get IEGLOutputStreamSettings Interface");
    printf("3Argus Version: %s\n", iCameraProvider->getVersion().c_str());
    iEGLStreamSettings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
    iEGLStreamSettings->setResolution(Size2D<uint32_t>(options.windowRect().width(),
                                                       options.windowRect().height()));
    iEGLStreamSettings->setEGLDisplay(g_display.get());
    printf("3Argus Version: %s\n", iCameraProvider->getVersion().c_str());
        stream[i] = iCaptureSession[i]->createOutputStream(streamSettings.get());
        iEGLOutputStream[i] = interface_cast<IEGLOutputStream>(stream[i]);
        EXIT_IF_NULL(iEGLOutputStream[i], "Cannot get IEGLOutputStream Interface");
    eglStreams.push_back(iEGLOutputStream[i]->getEGLStream());
/*        m_previewConsumerThread[i] = new PreviewConsumerThread(
            iEGLOutputStream[i]->getEGLDisplay(), iEGLOutputStream[i]->getEGLStream());
        PROPAGATE_ERROR(m_previewConsumerThread[i]->initialize());
        PROPAGATE_ERROR(m_previewConsumerThread[i]->waitRunning());
*/
        request[i] = iCaptureSession[i]->createRequest(CAPTURE_INTENT_PREVIEW);
        iRequest[i] = interface_cast<IRequest>(request[i]);
        EXIT_IF_NULL(iRequest[i], "Failed to get capture request interface");
    }
   m_previewConsumerThread[0] = new PreviewConsumerThread(g_display.get(), eglStreams,
                                         PreviewConsumerThread::LAYOUT_SPLIT_VERTICAL,
                                         false /* Sync stream frames */);
        PROPAGATE_ERROR(m_previewConsumerThread[0]->initialize());
        PROPAGATE_ERROR(m_previewConsumerThread[0]->waitRunning());
/*    
    appTearDown.m_stream = iSession->createOutputStream(streamSettings.get());

    IEGLOutputStream *iEGLOutputStream = interface_cast<IEGLOutputStream>(appTearDown.m_stream);
    EXIT_IF_NULL(iEGLOutputStream, "Cannot get IEGLOutputStream Interface");

    appTearDown.m_previewConsumerThread = new PreviewConsumerThread(
        iEGLOutputStream->getEGLDisplay(), iEGLOutputStream->getEGLStream());
    PROPAGATE_ERROR(appTearDown.m_previewConsumerThread->initialize());
    PROPAGATE_ERROR(appTearDown.m_previewConsumerThread->waitRunning());

    UniqueObj<Request> request(iSession->createRequest(CAPTURE_INTENT_PREVIEW));
    IRequest *iRequest = interface_cast<IRequest>(request);
    EXIT_IF_NULL(iRequest, "Failed to get capture request interface");


    // Set the sensor mode in the request.
    ISourceSettings *iSourceSettings =
        interface_cast<ISourceSettings>(iRequest->getSourceSettings());
    EXIT_IF_NULL(iSourceSettings, "Failed to get source settings interface");
    iSourceSettings->setSensorMode(sensorModes[0]);


    EXIT_IF_NOT_OK(iRequest->enableOutputStream(appTearDown.m_stream),
        "Failed to enable stream in capture request");
    */
    printf("3Argus Version: %s\n", iCameraProvider->getVersion().c_str());


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
        EXIT_IF_NOT_OK(iRequest[i]->enableOutputStream(stream[i]),
        "Failed to enable stream in capture request");
        EXIT_IF_NOT_OK(iCaptureSession[i]->repeat(request[i]), "Unable to submit repeat() request");
    }

/*
    IAutoControlSettings* iAutoControlSettings =
        interface_cast<IAutoControlSettings>(iRequest->getAutoControlSettings());
    EXIT_IF_NULL(iAutoControlSettings, "Failed to get AutoControlSettings interface");

    if (options.useAverageMap())
    {
        // Enable BayerAverageMap generation in the request.
        Ext::IBayerAverageMapSettings *iBayerAverageMapSettings =
            interface_cast<Ext::IBayerAverageMapSettings>(request);
        EXIT_IF_NULL(iBayerAverageMapSettings,
            "Failed to get BayerAverageMapSettings interface");
        iBayerAverageMapSettings->setBayerAverageMapEnable(true);
    }
*/
//    EXIT_IF_NOT_OK(iSession->repeat(request.get()), "Unable to submit repeat() request");

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
    while (frameCaptureLoop < 120)
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
/*
        iEventProvider->waitForEvents(queue.get(), FIVE_SECONDS);
        EXIT_IF_TRUE(iQueue->getSize() == 0, "No events in queue");

        const Event* event = iQueue->getEvent(iQueue->getSize() - 1);
        const IEvent* iEvent = interface_cast<const IEvent>(event);*/
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
                printf("EVENT_TYPE_CAPTURE_COMPLETE\n");
                //const TimeValue sensorTime = TimeValue::fromNSec(iMetadata->getSensorTimestamp());
                unsigned long long sensorTime = iMetadata->getSensorTimestamp();
                uint64_t frameduration = iMetadata->getFrameDuration();
                uint32_t captureId = iEvent->getCaptureId();
                if(captureId == 1)
                {
                    printf("set default duration %ld : %ld\n", orinframeduration, frameduration);
                    orinframeduration = frameduration;
               }
                //printf("cam1 id %d timestamps %.2f ms\n", iEvent->getCaptureId(), sensorTime/1000000.0f);

               if (iEvent1->getEventType() == EVENT_TYPE_CAPTURE_COMPLETE) {
                   const IEventCaptureComplete* iEventCaptureComplete1 =
                       interface_cast<const IEventCaptureComplete>(event1);
                   EXIT_IF_NULL(iEventCaptureComplete, "Failed to get EventCaptureComplete Interface");

                   const CaptureMetadata* metaData1 = iEventCaptureComplete1->getMetadata();
                   const ICaptureMetadata* iMetadata1 = interface_cast<const ICaptureMetadata>(metaData1);

                   EXIT_IF_NULL(iMetadata1, "Failed to get CaptureMetadata Interface");
                   printf("EVENT_TYPE_CAPTURE_COMPLETE\n");
                   //const TimeValue sensorTime = TimeValue::fromNSec(iMetadata->getSensorTimestamp());
                   unsigned long long sensorTime1 = iMetadata1->getSensorTimestamp();
                   uint64_t frameduration1 = iMetadata1->getFrameDuration();
                   uint32_t captureId1 = iEvent1->getCaptureId();
                   long long diff = labs(sensorTime1 - sensorTime);
                   printf("sensorTime %lld - %lld = %lld (%f)\n",
                       sensorTime,
                       sensorTime1,
                       diff,
                       sensorTime1/1000000.0f - sensorTime/1000000.0f);
                   //printf("cam2 id %d timestamps %.2f ms\n",iEvent1->getCaptureId() ,sensorTime1/1000000.0f);
                   printf("duration %ld isAdjust %d captrueId[%d, %d] timestamps [%.2f, %.2f]ms diff %llu frameduration [%ld, %ld]\n",
                       orinframeduration,
                       isAdjust,
                       captureId,
                       captureId1,
                       sensorTime/1000000.0f,
                       sensorTime1/1000000.0f,
                       diff,
                       frameduration,
                       frameduration1);
                  //iAutoControlSettings->setWbGains(bayerGains);
                  //iAutoControlSettings->setAwbMode(AWB_MODE_MANUAL);
                  if (isAdjust)
                  {
                      ISourceSettings *iSourceSettings =
                          interface_cast<ISourceSettings>(iRequest[adjustCamIndex]->getSourceSettings());
                      iSourceSettings->setFrameDurationRange(Range<uint64_t>(1e9/CAPTURE_FPS));
                      EXIT_IF_NOT_OK(iCaptureSession[adjustCamIndex]->repeat(request[adjustCamIndex]), "Unable to submit repeat() request");   
                      //#define NENOTOMILLI 1000000.0f
                      //#define threshold 5000
                      if(labs(orinframeduration - frameduration) < THRESHOLD)
                      {
                          isAdjust = false;
                          printf("-----------adjust timestamp back done-------------------------\n");
                      }
                      printf("adjust timestamp back\n");
                  }
                  if (diff > 10000 && !isAdjust)
                  {
                      adjustCamIndex = sensorTime1 > sensorTime ? 0 : 1;
                      ISourceSettings *iSourceSettings =
                          interface_cast<ISourceSettings>(iRequest[adjustCamIndex]->getSourceSettings());
                      iSourceSettings->setFrameDurationRange(Range<uint64_t>(1e9/CAPTURE_FPS + diff/2));
                      EXIT_IF_NOT_OK(iCaptureSession[adjustCamIndex]->repeat(request[adjustCamIndex]), "Unable to submit repeat() request");
                      isAdjust = true;
                      printf("adjust timestamp of Cam %d captrueId[%d, %d] timestamps [%.2f, %.2f]ms diff %llu\n",
                       adjustCamIndex,
                       captureId,
                       captureId1,
                       sensorTime/1000000.0f,
                       sensorTime1/1000000.0f,
                       diff);
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

    //iSession->stopRepeat is cleaned with captureSession RAII
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
