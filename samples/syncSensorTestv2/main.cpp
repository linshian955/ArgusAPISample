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
//add opts control
#include <getopt.h>

#define NENOTOMILLI 1000000.0f //ns to ms
#define DURATIONTOLRANCE 5000 //5us
#define FIVE_SECONDS 5000000000

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

/*******************************************************************************
 * Extended options class to add additional options specific to this sample.
 ******************************************************************************/
typedef struct
{
    uint32_t sensor_mode;
    int selectedFPS;
    int countsForCaptureLoop;
    int captrueFameNumber;
    int width;
    int height;
    bool ready;
    long long syncThreshold;
    int outputType;
} InputOptions;

class CameraInfo
{
public:
    //options
    uint32_t choosedSensorMode;
    int previewFPS;
    int height;
    int width;
    int outputType;

    CameraProvider* m_cameraProvider;
    //sensor devices
    std::vector<CameraDevice*> cameraDevices;

    //sensor mode
    std::vector<SensorMode*> sensorModes;

    //eventTypes
    std::vector<EventType> eventTypes;

    IEventProvider *iEventProvider[MAX_CAM_DEVICE];
    EventQueue *queue[MAX_CAM_DEVICE];    
    IEventQueue *iQueue[MAX_CAM_DEVICE];

    //capture sessions
    CaptureSession* captureSession[MAX_CAM_DEVICE];
    ICaptureSession* iCaptureSession[MAX_CAM_DEVICE];
    
    UniqueObj<OutputStreamSettings> streamSettings;

    //preview streams, threads, requests    
    Request *previewRequest[MAX_CAM_DEVICE];
    IRequest *iPreviewRequest[MAX_CAM_DEVICE];

    
    OutputStream *previewStream[MAX_CAM_DEVICE];
    IEGLOutputStream *iEGLOutputStream[MAX_CAM_DEVICE];   
    PreviewConsumerThread* m_previewConsumerThread;

    //capture streams, threads, requests 
    Request *captureRequest[MAX_CAM_DEVICE];
    IRequest *iCaptureRequest[MAX_CAM_DEVICE];

    OutputStream *capture_stream[MAX_CAM_DEVICE];

    CaptureConsumerThread* m_captureConsumer[MAX_CAM_DEVICE];

    CameraInfo()
    {
        m_cameraProvider = CameraProvider::create();
        getICameraProvider()->getCameraDevices(&cameraDevices);
        for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++) {
            m_captureConsumer[i] = NULL;
            previewStream[i] = NULL;
        }
        m_previewConsumerThread = NULL;
    }
    CameraInfo(uint32_t sensormode,int fps, int w, int h, int output)
        : choosedSensorMode(sensormode), previewFPS(fps), width(w), height(h), outputType(output)
    {
        init();
    }
    bool init()
    {
        std::vector<EGLStreamKHR> eglStreams;
        m_cameraProvider = CameraProvider::create();
        getICameraProvider()->getCameraDevices(&cameraDevices);
        getSensorModes();
        initEventTypes();
        
        for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++) {
            //init CaptureSession
            captureSession[i] = getICameraProvider()->createCaptureSession(cameraDevices[i]);
            iCaptureSession[i] = interface_cast<ICaptureSession>(captureSession[i]);
            EXIT_IF_NULL(iCaptureSession[i], "Cannot get Capture Session Interface");

            //init Event Interface
            iEventProvider[i] = interface_cast<IEventProvider>(captureSession[i]);
            EXIT_IF_NULL(iEventProvider[i], "iEventProvider is NULL");

            queue[i] = iEventProvider[i]->createEventQueue(eventTypes);
            iQueue[i] = interface_cast<IEventQueue>(queue[i]);
            EXIT_IF_NULL(iQueue[i], "event queue interface is NULL");

            //contruct streams

            //init StreamSettings
            UniqueObj<OutputStreamSettings> streamSettings(iCaptureSession[i]->createOutputStreamSettings(STREAM_TYPE_EGL));
            IEGLOutputStreamSettings *iEGLStreamSettings = interface_cast<IEGLOutputStreamSettings>(streamSettings);
            EXIT_IF_NULL(iEGLStreamSettings, "Cannot get IEGLOutputStreamSettings Interface");

            iEGLStreamSettings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
            /*
            iEGLStreamSettings->setResolution(Size2D<uint32_t>(options.windowRect().width(),
                                                       options.windowRect().height()));*/
            iEGLStreamSettings->setResolution(Size2D<uint32_t>(width,
                                                       height));
            iEGLStreamSettings->setEGLDisplay(g_display.get());
            iEGLStreamSettings->setMetadataEnable(true);

            //config preview stream to capturesession
            previewStream[i] = iCaptureSession[i]->createOutputStream(streamSettings.get());
            iEGLOutputStream[i] = interface_cast<IEGLOutputStream>(previewStream[i]);
            EXIT_IF_NULL(iEGLOutputStream[i], "Cannot get IEGLOutputStream Interface");
            
            //keep eglStream instance of preview stream for preview thread
            eglStreams.push_back(iEGLOutputStream[i]->getEGLStream());

            //config captrue stream to capturesession
            capture_stream[i] = iCaptureSession[i]->createOutputStream(streamSettings.get());
            EXIT_IF_NULL(capture_stream[i], "Failed to create EGLOutputStream");

            //init request of preview and capture
            previewRequest[i] = iCaptureSession[i]->createRequest(CAPTURE_INTENT_PREVIEW);
            iPreviewRequest[i] = interface_cast<IRequest>(previewRequest[i]);
            EXIT_IF_NULL(iPreviewRequest[i], "Failed to get capture request interface");

            captureRequest[i] = iCaptureSession[i]->createRequest(CAPTURE_INTENT_STILL_CAPTURE);
            iCaptureRequest[i] = interface_cast<IRequest>(captureRequest[i]);
            EXIT_IF_NULL(iCaptureRequest[i], "Failed to get capture request interface");

            //init thread of capture stream
            m_captureConsumer[i] = new CaptureConsumerThread(capture_stream[i],i,outputType);
            PROPAGATE_ERROR(m_captureConsumer[i]->initialize());
            PROPAGATE_ERROR(m_captureConsumer[i]->waitRunning());
        }
        //init thread of preview stream
        m_previewConsumerThread = new PreviewConsumerThread(g_display.get(), eglStreams,
                                        PreviewConsumerThread::LAYOUT_SPLIT_VERTICAL,
                                        false );
        PROPAGATE_ERROR(m_previewConsumerThread->initialize());
        PROPAGATE_ERROR(m_previewConsumerThread->waitRunning());
        
        startStreams();
        return true;
    }
    
    bool startStreams()
    {
        if (choosedSensorMode >= sensorModes.size())
            ORIGINATE_ERROR("Sensor mode index is out of range");

        SensorMode *sensorMode = sensorModes[choosedSensorMode];
        for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++) {
			//preview streams
            ISourceSettings *iSourceSettings = interface_cast<ISourceSettings>(iPreviewRequest[i]->getSourceSettings());
            iSourceSettings->setSensorMode(sensorMode);
            iSourceSettings->setFrameDurationRange(Range<uint64_t>(1e9/previewFPS));
            EXIT_IF_NOT_OK(iPreviewRequest[i]->enableOutputStream(previewStream[i]),"Failed to enable stream in capture request");
            EXIT_IF_NOT_OK(iCaptureSession[i]->repeat(previewRequest[i]), "Unable to submit repeat() request");
            //capture streams
            iSourceSettings = interface_cast<ISourceSettings>(iCaptureRequest[i]->getSourceSettings());
            iSourceSettings->setSensorMode(sensorMode);
            iSourceSettings->setFrameDurationRange(Range<uint64_t>(1e9/previewFPS));
            EXIT_IF_NOT_OK(iCaptureRequest[i]->enableOutputStream(capture_stream[i]),"Failed to enable stream in capture request");
        }
        return true;
    }

    ICameraProvider* getICameraProvider()
    {
        return interface_cast<ICameraProvider>(m_cameraProvider);
    }
	
    bool getSensorModes()
    {
        if (cameraDevices.size() < 2)
            ORIGINATE_ERROR("Must have at least 2 sensors available");

        ICameraProperties *iCameraProperties = interface_cast<ICameraProperties>(cameraDevices[0]);
        if (!iCameraProperties)
            ORIGINATE_ERROR("Failed to get ICameraProperties interface");

        ISensorMode *iSensorMode;
        iCameraProperties->getBasicSensorModes(&sensorModes);
        if (sensorModes.size() == 0)
            ORIGINATE_ERROR("Failed to get sensor modes");

        printf("Available Sensor modes :\n");
        for (uint32_t i = 0; i < sensorModes.size(); i++)
        {
            iSensorMode = interface_cast<ISensorMode>(sensorModes[i]);
            Size2D<uint32_t> resolution = iSensorMode->getResolution();
            printf("[%u] W=%u H=%u\n", i, resolution.width(), resolution.height());
        }
        return true;
    }

    void initEventTypes()
    {
        // Create the CaptureSession using the selected device.
        eventTypes.push_back(EVENT_TYPE_CAPTURE_COMPLETE);
        eventTypes.push_back(EVENT_TYPE_ERROR);
        /* Seems there is bug in Argus, which drops EVENT_TYPE_ERROR if all
        3 events are not set. Set it for now */
        eventTypes.push_back(EVENT_TYPE_CAPTURE_STARTED);
    }

    void shutdown()
    {
        for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++) {
            iCaptureSession[i]->stopRepeat();
            iCaptureSession[i]->waitForIdle();
            if (m_captureConsumer[i] != NULL)
            {
                PROPAGATE_ERROR_CONTINUE(m_captureConsumer[i]->shutdown());
                delete m_captureConsumer[i];
                m_captureConsumer[i] = NULL;
            }
            if(capture_stream[i] != NULL)
                capture_stream[i]->destroy();
            if(previewStream[i] != NULL)
                previewStream[i]->destroy();

        }
        if (m_previewConsumerThread!= NULL)
        {
            PROPAGATE_ERROR_CONTINUE(m_previewConsumerThread->shutdown());
            delete m_previewConsumerThread;
            m_previewConsumerThread = NULL;
        }
        if (m_cameraProvider != NULL)
            m_cameraProvider->destroy();
        Window::getInstance().shutdown();
        PROPAGATE_ERROR_CONTINUE(g_display.cleanup());
    }

    ~CameraInfo()
    {
        shutdown();
    }
};


/*
 * Program: userAutoWhiteBalance
 * Function: To display 101 preview images to the device display to illustrate a
 *           Grey World White Balance technique for adjusting the White Balance in real time
 *           through the use of setWbGains() and setAwbMode(AWB_MODE_MANUAL) calls.
 */

static bool execute(InputOptions& options)
{
    static const Argus::Rectangle<uint32_t> DEFAULT_WINDOW_RECT(0, 0, 1024, 768);
    // Initialize the window and EGL display.
    Window::getInstance().setWindowRect(DEFAULT_WINDOW_RECT);
    PROPAGATE_ERROR(g_display.initialize(Window::getInstance().getEGLNativeDisplay()));

    /*
     * Set up Argus API Framework, identify available camera devices, create
     * a capture session for the first available device, and set up the event
     * queue for completed events
     */

    CameraInfo cameraSyncInfo(options.sensor_mode, options.selectedFPS, options.width, options.height, options.outputType);
    
    uint64_t orinframeduration = 1e9/options.selectedFPS;

    uint32_t frameCaptureLoop = 0;
    bool isAdjust = false;
    int adjustCamIndex = 0;
    long long diff;
    Argus::Status status;
    const IEventCaptureComplete* iEventCaptureComplete;
    const CaptureMetadata* metaData;
    const ICaptureMetadata* iMetadata;

    unsigned long long sensorTime[2];
    uint32_t captureId[2];
    uint64_t frameduration[2];
    const Event* event[2];
    const IEvent* iEvent[2];

    while (frameCaptureLoop < options.countsForCaptureLoop)
    {
        // Keep PREVIEW display window serviced
        Window::getInstance().pollEvents();

        cameraSyncInfo.iEventProvider[0]->waitForEvents(cameraSyncInfo.queue[0], FIVE_SECONDS);
        EXIT_IF_TRUE(cameraSyncInfo.iQueue[0]->getSize() == 0, "No events in queue");
        event[0] = cameraSyncInfo.iQueue[0]->getEvent(cameraSyncInfo.iQueue[0]->getSize() - 1);
        iEvent[0] = interface_cast<const IEvent>(event[0]);
        
        cameraSyncInfo.iEventProvider[1]->waitForEvents(cameraSyncInfo.queue[1], FIVE_SECONDS);
        EXIT_IF_TRUE(cameraSyncInfo.iQueue[1]->getSize() == 0, "No events in queue");
        event[1] = cameraSyncInfo.iQueue[1]->getEvent(cameraSyncInfo.iQueue[1]->getSize() - 1);
        iEvent[1] = interface_cast<const IEvent>(event[1]);
        if(!iEvent[0])
        {
            printf("Failed to get IEvent[0] interface");
        }
        else
        {
            if (iEvent[0]->getEventType() == EVENT_TYPE_CAPTURE_COMPLETE) {
                frameCaptureLoop++;
                
                //check event contents
                iEventCaptureComplete = interface_cast<const IEventCaptureComplete>(event[0]);
                EXIT_IF_NULL(iEventCaptureComplete, "Failed to get EventCaptureComplete Interface");
                metaData = iEventCaptureComplete->getMetadata();
                iMetadata = interface_cast<const ICaptureMetadata>(metaData);
                EXIT_IF_NULL(iMetadata, "Failed to get CaptureMetadata Interface");                
                printf("iEvent[0] EVENT_TYPE_CAPTURE_COMPLETE\n");
                
                //get metadata info
                sensorTime[0] = iMetadata->getSensorTimestamp();
                frameduration[0] = iMetadata->getFrameDuration();
                captureId[0] = iEvent[0]->getCaptureId();
                
                EXIT_IF_NULL(iEvent[1], "Failed to get IEvent[1] interface");
                if (iEvent[1]->getEventType() == EVENT_TYPE_CAPTURE_COMPLETE) {
                   //check event contents
                    iEventCaptureComplete = interface_cast<const IEventCaptureComplete>(event[1]);
                    EXIT_IF_NULL(iEventCaptureComplete, "Failed to get EventCaptureComplete Interface");
                    metaData = iEventCaptureComplete->getMetadata();
                    iMetadata = interface_cast<const ICaptureMetadata>(metaData);
                    EXIT_IF_NULL(iMetadata, "Failed to get CaptureMetadata Interface");                    
                    printf("iEvent[1] EVENT_TYPE_CAPTURE_COMPLETE\n");

                    //get metadata info
                    sensorTime[1] = iMetadata->getSensorTimestamp();
                    frameduration[1] = iMetadata->getFrameDuration();
                    captureId[1] = iEvent[1]->getCaptureId();
                    
                    //compute timstamp difference
                    diff = labs(sensorTime[1] - sensorTime[0]);
                    printf("SensorTimeStamp diff: %lld - %lld = %lld (%f)\n",
                        sensorTime[0],
                        sensorTime[1],
                        diff,
                        sensorTime[1]/NENOTOMILLI - sensorTime[0]/NENOTOMILLI);

                    printf("StreamDuration %ld isAdjust %d CaptrueId[%d, %d] TimeStamps [%.2f, %.2f]ms diff %llu Frameduration [%ld, %ld]\n",
                        orinframeduration,
                        isAdjust,
                        captureId[0],
                        captureId[1],
                        sensorTime[0]/NENOTOMILLI,
                        sensorTime[1]/NENOTOMILLI,
                        diff,
                        frameduration[0],
                        frameduration[1]);

                    //algorithm to sync sensor by adjust frame duration
                    if (isAdjust)
                    {
                        ISourceSettings *iSourceSettings =
                            interface_cast<ISourceSettings>(cameraSyncInfo.iPreviewRequest[adjustCamIndex]->getSourceSettings());
                        iSourceSettings->setFrameDurationRange(Range<uint64_t>(orinframeduration));
                        EXIT_IF_NOT_OK(cameraSyncInfo.iCaptureSession[adjustCamIndex]->repeat(cameraSyncInfo.previewRequest[adjustCamIndex]), "Unable to submit repeat() request");   

                        if(labs(orinframeduration - frameduration[adjustCamIndex]) < DURATIONTOLRANCE)
                        {
                            isAdjust = false;
                            printf("-----------stop set timestamp back-------------------------\n");
                        }
                        printf("try to set TimeStamp back\n");
                    }
                    if (diff > options.syncThreshold && !isAdjust)
                    {
                        adjustCamIndex = sensorTime[1] > sensorTime[0] ? 0 : 1;
                        ISourceSettings *iSourceSettings =
                            interface_cast<ISourceSettings>(cameraSyncInfo.iPreviewRequest[adjustCamIndex]->getSourceSettings());
                        iSourceSettings->setFrameDurationRange(Range<uint64_t>(orinframeduration + diff/2));
                        EXIT_IF_NOT_OK(cameraSyncInfo.iCaptureSession[adjustCamIndex]->repeat(cameraSyncInfo.previewRequest[adjustCamIndex]), "Unable to submit repeat() request");
                        isAdjust = true;
                        printf("adjust timestamp of Cam %d captrueId[%d, %d] timestamps [%.2f, %.2f]ms diff %.2f\n",
                            adjustCamIndex,
                            captureId[0],
                            captureId[1],
                            sensorTime[0]/NENOTOMILLI,
                            sensorTime[1]/NENOTOMILLI,
                            diff/NENOTOMILLI);
                    }
                    else if (diff < options.syncThreshold && !isAdjust && captureId[0] > options.captrueFameNumber)
                    {
                        printf("dump image of isAdjust %d captrueId[%d, %d] timestamps [%.2f, %.2f]ms diff %.2f\n",
                            isAdjust,
                            captureId[0],
                            captureId[1],
                            sensorTime[0]/NENOTOMILLI,
                            sensorTime[1]/NENOTOMILLI,
                            diff/NENOTOMILLI);
		                break;
                     }
               }
            } else if (iEvent[0]->getEventType() == EVENT_TYPE_CAPTURE_STARTED) {
                /* ToDo: Remove the empty after the bug is fixed */
                continue;
            } else if (iEvent[0]->getEventType() == EVENT_TYPE_ERROR) {
                const IEventError* iEventError =
                    interface_cast<const IEventError>(event[0]);
                EXIT_IF_NOT_OK(iEventError->getStatus(), "ERROR event");
            } else {
                printf("WARNING: Unknown event. Continue\n");
            }
        }
    }
    if(frameCaptureLoop < options.countsForCaptureLoop)
    {
        cameraSyncInfo.iCaptureSession[0]->capture(cameraSyncInfo.captureRequest[0]);
        cameraSyncInfo.iCaptureSession[1]->capture(cameraSyncInfo.captureRequest[1]);
        sleep(3);
    }
    
    //cameraSyncInfo.shutdown();
    sleep(2);
    return true;
}

}; // namespace ArgusSamples

int main(int argc, char **argv)
{
	//default options
    ArgusSamples::InputOptions options;
    options.sensor_mode = 1;
    options.selectedFPS = 30;
    options.countsForCaptureLoop = 120;
    options.captrueFameNumber = 100;
    options.width = 3280;
    options.height = 2464;
    options.syncThreshold = 10*1e6;
    options.outputType = 0;
    options.ready = true;
    static std::string optstring = "m:f:t:c:w:e:s:o:h";
    static int option_index;
    static struct option long_opts[] {
        {"sensorMode", optional_argument, NULL, 'm'},
        {"FPS", optional_argument, NULL, 'f'},
        {"totalFrameCount", optional_argument, NULL, 't'},
        {"captureFrameCount", optional_argument, NULL, 'c'},
        {"width", optional_argument, NULL, 'w'},
        {"height", optional_argument, NULL, 'h'},
        {"syncThreshold", optional_argument, NULL, 's'},
        {"ouputType", optional_argument, NULL, 'o'},
        {"help", no_argument, NULL, 'e'},
        {NULL , 0, NULL, 0}
    };
    while(1) {
        int c;
        c = getopt_long(argc, argv, optstring.c_str(), long_opts, &option_index);
        
        if(c == -1)
            break;

        switch(c)
        {
            case 'm':
                options.sensor_mode = atoi(optarg);
                break;
            case 'f':
                options.selectedFPS = atoi(optarg);
                printf("fps %d",atoi(optarg));
                break;
            case 't':
                options.countsForCaptureLoop = atoi(optarg);
                break;
            case 'c':
                options.captrueFameNumber = atoi(optarg);
                break;
            case 'w':
                options.width = atoi(optarg);
                break;
            case 'e':
                options.height = atoi(optarg);
                break;
            case 's':
                options.syncThreshold = atoi(optarg)*1e6;
                break;
            case 'o':
                options.outputType = atoi(optarg);
            case 'h':
            default:
                ArgusSamples::CameraInfo cameraSyncInfo;
                printf("----help----\n");
                printf("sensorMode(-s): choose sensor support modes showed following\n");
                cameraSyncInfo.getSensorModes();
                printf("FPS(-s): fps for stream config e.g 30 \n");
                printf("totalFrameCount(-t): The Test will keep runing after received totalFrameCount frames e.g 120\n");
                printf("captureFrameCount(-c): tigger caputre when frame number is over captureFrameCount and sensor in sync e.g 100\n");
                printf("width(-w): width of capture stream e.g 1920\n");
                printf("height(-e): height of capture stream e.g 1080\n");
                printf("syncThreshold(-s): the threshold(ms) to check cameras is synced e.g 10\n");
                printf("----help----\n");
                options.ready = false;
                break;
        }
    }
    if(options.ready)
    {
        printf("Input arg:\n sensor_mode %d\n fps %d\n loopCounts %d\n captureFrameNum %d\n wxd %dx%d\n",
            options.sensor_mode,
            options.selectedFPS,
            options.countsForCaptureLoop,
            options.captrueFameNumber,
            options.width,
            options.height);
        if (!ArgusSamples::execute(options))
        {
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
