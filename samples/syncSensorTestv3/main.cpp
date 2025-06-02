/*
 * Copyright (c) 2016 - 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Argus/Argus.h>
#include <cuda.h>
#include <cudaEGL.h>

#include "CommonOptions.h"
#include "CUDAHelper.h"
#include "EGLGlobal.h"
#include "Error.h"
#include "KLDistance.h"
#include "Thread.h"

// Histogram generation methods borrowed from cudaHistogram sample.
#include "../cudaHistogram/histogram.h"

#include <Argus/Ext/SensorTimestampTsc.h>
#include <EGLStream/EGLStream.h>
#define SYNC_THRESHOLD_TIME_US 100.0f
#define FPS 30
#define NENOSECOND_TO_MILLISECOND 1000000.0f //ns to ms
#define DURATION_CONTROL_DELAY 5
#define SYNCTHREHOLD 1*1e6
enum maxCamDevice
{
    LEFT_CAM_DEVICE  = 0,
    RIGHT_CAM_DEVICE = 1,
    MAX_CAM_DEVICE = 2
};
using namespace Argus;
using namespace EGLStream;
#define FILE_DIR "/home/asus/Desktop/result/"
namespace ArgusSamples
{

/*
 * This sample opens a session with two sensors, it then using CUDA computes the histogram
 * of the sensors and computes a KL distance between the two histograms. A small value near
 * 0 indicates that the two images are alike. The processing of the images happens in the worker
 * thread of StereoDisparityConsumerThread. While the main app thread is used to drive the captures.
 */

// Constants.
static const Size2D<uint32_t> STREAM_SIZE(640, 480);

// Globals and derived constants.
EGLDisplayHolder g_display;

// Debug print macros.
#define PRODUCER_PRINT(...) printf("PRODUCER: " __VA_ARGS__)
#define CONSUMER_PRINT(...) printf("CONSUMER: " __VA_ARGS__)

#define EXIT_IF_TRUE(val,msg)   \
        {if ((val)) {printf("%s\n",msg); return false;}}
#define EXIT_IF_NULL(val,msg)   \
        {if (!val) {printf("%s\n",msg); return false;}}
#define EXIT_IF_NOT_OK(val,msg) \
        {if (val!=Argus::STATUS_OK) {printf("%s\n",msg); return false;}}
/*******************************************************************************
 * Argus disparity class
 *   This class will analyze frames from 2 synchronized sensors and compute the
 *   KL distance between the two images. Large values of KL indicate a large disparity
 *   while a value of 0.0 indicates that the images are alike.
 ******************************************************************************/
class StereoDisparityConsumerThread : public Thread
{
public:
    explicit StereoDisparityConsumerThread(ICaptureSession **iSession,
                                           OutputStream **pStream,
                                           Request ** pRequest)
    {
        for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++)
        {
            iCaptureSession[i] = iSession[i];
            previewStream[i] = pStream[i];
            previewRequest[i] = pRequest[i];
            printf("session %p,stream %p, request %p \n", &iSession[i], &pStream[i], &pRequest[i]);
        }
    }
    ~StereoDisparityConsumerThread()
    {
        for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++)
        {
            if(iFrame[i])
                iFrame[i]->releaseFrame();
        }
    }
private:
    /** @name Thread methods */
    /**@{*/
    virtual bool threadInitialize();
    virtual bool threadExecute();
    virtual bool threadShutdown();
    /**@}*/

   //capture sessions
    ICaptureSession* iCaptureSession[MAX_CAM_DEVICE];

    //preview streams, threads, requests    
    Request *previewRequest[MAX_CAM_DEVICE];
    IRequest *iPreviewRequest[MAX_CAM_DEVICE];

    OutputStream *previewStream[MAX_CAM_DEVICE];
    IEGLOutputStream *iEGLOutputStream[MAX_CAM_DEVICE];
    UniqueObj<FrameConsumer> m_Consumer[MAX_CAM_DEVICE];

    IFrameConsumer* iFrameConsumer[MAX_CAM_DEVICE];
    CaptureMetadata* captureMetadata[MAX_CAM_DEVICE];
    ICaptureMetadata* iMetadata[MAX_CAM_DEVICE];
    
    Frame *frame[MAX_CAM_DEVICE];
    IFrame *iFrame[MAX_CAM_DEVICE];
    EGLStream::Image *image[MAX_CAM_DEVICE];
    
    Ext::ISensorTimestampTsc *iSensorTimestampTsc[MAX_CAM_DEVICE];
    unsigned long long tscTimeStamp[MAX_CAM_DEVICE];
    
    unsigned long long frameNumber[MAX_CAM_DEVICE];
    uint64_t frameduration[MAX_CAM_DEVICE];
};


bool StereoDisparityConsumerThread::threadInitialize()
{
    for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++)
    {
        iEGLOutputStream[i] = interface_cast<IEGLOutputStream>(previewStream[i]);
        if(iEGLOutputStream[i])
        {
            m_Consumer[i] = UniqueObj<FrameConsumer>(FrameConsumer::create(previewStream[i]));
            if (!m_Consumer[i])
                ORIGINATE_ERROR("Failed to create FrameConsumer for m_Consumer[%d] stream", i);
        }
        printf("iEGLOutputStream %p,previewStream %p, m_Consumer %p \n", &iEGLOutputStream[i], &previewStream[i], &m_Consumer[i]);

        iPreviewRequest[i] = interface_cast<IRequest>(previewRequest[i]);
        EXIT_IF_NULL(iPreviewRequest[i], "Failed to get capture request interface");
    }
    return true;
}

bool StereoDisparityConsumerThread::threadExecute()
{
    char filepath[80];
    unsigned long long diff = 0;
    int adjustCamIndex = 0;
    unsigned long long preAsyncFrameNumber = -DURATION_CONTROL_DELAY;
    unsigned long long step = 0;
    bool isSync = false;
    uint64_t orinframeduration = 1e9/FPS;

    for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++) {

        iFrameConsumer[i] = interface_cast<IFrameConsumer>(m_Consumer[i]);
        if (!iFrameConsumer[i])
        {
            ORIGINATE_ERROR("iFrameConsumer[%d]: Failed to get stream cosumer\n",i);
        }
        // Wait until the producer has connected to the stream.
        CONSUMER_PRINT("iEGLOutputStream[%d]: Waiting until Argus producer is connected to stream...\n",
            i);
        if (iEGLOutputStream[i]->waitUntilConnected() != STATUS_OK)
            ORIGINATE_ERROR("Argus producer failed to connect to right stream.");
        CONSUMER_PRINT("iEGLOutputStream[%d]: Argus producer has connected; continuing.\n", i);
    }
    
    while (true)
    {
        for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++)
	    {
			//get frame
            frame[i] = iFrameConsumer[i]->acquireFrame();
            iFrame[i] = interface_cast<IFrame>(frame[i]);

            if (!iFrame[i])
               continue;

            //get metadata
            captureMetadata[i] = interface_cast<IArgusCaptureMetadata>(frame[i])->getMetadata();
            iMetadata[i] = interface_cast<ICaptureMetadata>(captureMetadata[i]);
            if (!captureMetadata || !iMetadata)
                ORIGINATE_ERROR("Cannot get metadata for frame of cam %d", i);

            /*
            // seems only works when two camera assign to one capture session
            if (iMetadata[i]->getSourceIndex() != i)
                ORIGINATE_ERROR("Incorrect sensor connected to stream %d %d", i,iMetadata[i]->getSourceIndex());
            */

            //get timestamp interface
            iSensorTimestampTsc[i] = interface_cast<Ext::ISensorTimestampTsc>(captureMetadata[i]);
            if (!iSensorTimestampTsc[i])
                ORIGINATE_ERROR("failed to get iSensorTimestampTsc[%d] inteface", i);

            //record related infomations
            tscTimeStamp[i] = iSensorTimestampTsc[i]->getSensorSofTimestampTsc();
            frameNumber[i] = iFrame[i]->getNumber();
            image[i] = iFrame[i]->getImage();
            frameduration[i] = iMetadata[i]->getFrameDuration();
            printf("Camera[%d]FrameNumber[%lld]Timestamp[%lld]Frameduration[%ld]\n",
                i,
                frameNumber[i],
                tscTimeStamp[i],
                frameduration[i]);
        }
        if(!frame[1] || !frame[0])
        {
            break;
        }
        printf("Timestamp dfferent calculation start\n");
        diff = labs(tscTimeStamp[1] - tscTimeStamp[0]);
        if (isSync)
        {
            ISourceSettings *iSourceSettings =
                interface_cast<ISourceSettings>(iPreviewRequest[adjustCamIndex]->getSourceSettings());
            iSourceSettings->setFrameDurationRange(Range<uint64_t>(orinframeduration));
            EXIT_IF_NOT_OK(iCaptureSession[adjustCamIndex]->repeat(previewRequest[adjustCamIndex]), "Unable to submit repeat() request");
            isSync = false;
            printf("\n\nFrameNumber[%lld, %lld]Timestamps(ms)[%.2f, %.2f]Diff[%.2f] SetFrameDurationRange back to stream config of Cam[%d] \n\n",
                        frameNumber[0],
                        frameNumber[1],
                        tscTimeStamp[0]/NENOSECOND_TO_MILLISECOND,
                        tscTimeStamp[1]/NENOSECOND_TO_MILLISECOND,
                        diff/NENOSECOND_TO_MILLISECOND,
                        adjustCamIndex);
        }
        if (diff > SYNCTHREHOLD && !isSync && frameNumber[0] - preAsyncFrameNumber > DURATION_CONTROL_DELAY)
        {
            //get related infomations
            adjustCamIndex = tscTimeStamp[1] > tscTimeStamp[0] ? 0 : 1;
            preAsyncFrameNumber = frameNumber[adjustCamIndex];
            step = diff < orinframeduration/2 ? diff : orinframeduration/2;
            
            //modify frame duration
            ISourceSettings *iSourceSettings =
                interface_cast<ISourceSettings>(iPreviewRequest[adjustCamIndex]->getSourceSettings());
            iSourceSettings->setFrameDurationRange(Range<uint64_t>(orinframeduration + step));
            EXIT_IF_NOT_OK(iCaptureSession[adjustCamIndex]->repeat(previewRequest[adjustCamIndex]), "Unable to submit repeat() request");

            isSync = true;
            
            printf("\n\nFrameNumber[%lld, %lld]Timestamps(ms)[%.2f, %.2f]Diff[%.2f] set Camera[%d] FrameDurationRange as %lld \n\n",
                            frameNumber[0],
                            frameNumber[1],
                            tscTimeStamp[0]/NENOSECOND_TO_MILLISECOND,
                            tscTimeStamp[1]/NENOSECOND_TO_MILLISECOND,
                            diff/NENOSECOND_TO_MILLISECOND,
                            adjustCamIndex,
                            orinframeduration + step);
         }
         else if (diff < SYNCTHREHOLD && !isSync)
         {
             printf("\n\nFrameNumber[%lld, %lld]Timestamps(ms)[%.2f, %.2f]Diff[%.2f] isSync %d Dump image\n\n",
                        frameNumber[0],
                        frameNumber[1],
                        tscTimeStamp[0]/NENOSECOND_TO_MILLISECOND,
                        tscTimeStamp[1]/NENOSECOND_TO_MILLISECOND,
                        diff/NENOSECOND_TO_MILLISECOND,
                        isSync);
             for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++)
             {
                 if(!image[i])
	             {
                     ORIGINATE_ERROR("Failed to get Image from iFrame->getImage()");
                 }
                 
		         EGLStream::IImageJPEG *iImageJPEG = Argus::interface_cast<EGLStream::IImageJPEG>(image[i]);
	             if(!iImageJPEG)
	             {
	                 ORIGINATE_ERROR("Failed to get ImageJPEG Interface");
                 }
	             sprintf(filepath,FILE_DIR "[T%.2f][FNum%llu]Cam[%d][D%ld].jpg",
	                 tscTimeStamp[i]/NENOSECOND_TO_MILLISECOND,
	                 frameNumber[i],
	                 i,
	                 frameduration[i]);

                 Argus::Status status = iImageJPEG->writeJPEG(filepath);
             }
         }
         for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++)
         {
             if(iFrame[i])
                 iFrame[i]->releaseFrame();
         }
     }
    CONSUMER_PRINT("No more frames. Cleaning up.\n");

    PROPAGATE_ERROR(requestShutdown());

    return true;
}

bool StereoDisparityConsumerThread::threadShutdown()
{
    CONSUMER_PRINT("threadShutdown----------\n");
    return true;
}



static bool execute(const CommonOptions& options)
{
    // Initialize EGL.
    PROPAGATE_ERROR(g_display.initialize());

    // Initialize the Argus camera provider.
    UniqueObj<CameraProvider> cameraProvider(CameraProvider::create());

    // Get the ICameraProvider interface from the global CameraProvider.
    ICameraProvider *iCameraProvider = interface_cast<ICameraProvider>(cameraProvider);
    if (!iCameraProvider)
        ORIGINATE_ERROR("Failed to get ICameraProvider interface");
    printf("Argus Version: %s\n", iCameraProvider->getVersion().c_str());

    // Get the camera devices.
    std::vector<CameraDevice*> cameraDevices;
    iCameraProvider->getCameraDevices(&cameraDevices);
    if (cameraDevices.size() < 2)
        ORIGINATE_ERROR("Must have at least 2 sensors available");

    //Get supported sensor modes
    ICameraProperties *iCameraProperties = interface_cast<ICameraProperties>(cameraDevices[1]);
    if (!iCameraProperties)
        ORIGINATE_ERROR("Failed to get ICameraProperties interface");

    std::vector<SensorMode*> sensorModes;
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

   //capture sessions
    CaptureSession* captureSession[MAX_CAM_DEVICE] = {NULL};
    ICaptureSession* iCaptureSession[MAX_CAM_DEVICE] = {NULL};
    
    OutputStreamSettings *streamSettings[MAX_CAM_DEVICE] = {NULL};

    //preview streams, threads, requests    
    Request *previewRequest[MAX_CAM_DEVICE] = {NULL};
    IRequest *iPreviewRequest[MAX_CAM_DEVICE] = {NULL};

    OutputStream *previewStream[MAX_CAM_DEVICE] = {NULL};

    for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++) {
        //init CaptureSession
        captureSession[i] = iCameraProvider->createCaptureSession(cameraDevices[i]);
        iCaptureSession[i] = interface_cast<ICaptureSession>(captureSession[i]);
        EXIT_IF_NULL(iCaptureSession[i], "Cannot get Capture Session Interface");

        //contruct streams

        //init StreamSettings
        streamSettings[i] = iCaptureSession[i]->createOutputStreamSettings(STREAM_TYPE_EGL);
        //setting if two camera in one camera session, default set as first camera in camera session
        IOutputStreamSettings *iStreamSettings = interface_cast<IOutputStreamSettings>(streamSettings[i]);
        iStreamSettings->setCameraDevice(cameraDevices[i]);
        IEGLOutputStreamSettings *iEGLStreamSettings = interface_cast<IEGLOutputStreamSettings>(streamSettings[i]);
        EXIT_IF_NULL(iEGLStreamSettings, "Cannot get IEGLOutputStreamSettings Interface");

        iEGLStreamSettings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);

        iEGLStreamSettings->setResolution(STREAM_SIZE);
        iEGLStreamSettings->setEGLDisplay(g_display.get());
        iEGLStreamSettings->setMetadataEnable(true);

        //config preview stream to capturesession
        previewStream[i] = iCaptureSession[i]->createOutputStream(streamSettings[i]);

        //init request of preview and capture
        previewRequest[i] = iCaptureSession[i]->createRequest(CAPTURE_INTENT_PREVIEW);
        iPreviewRequest[i] = interface_cast<IRequest>(previewRequest[i]);
        EXIT_IF_NULL(iPreviewRequest[i], "Failed to get capture request interface");
        
    }

    PRODUCER_PRINT("Launching disparity checking consumer\n");
    StereoDisparityConsumerThread disparityConsumer(iCaptureSession, previewStream, previewRequest);
    PROPAGATE_ERROR(disparityConsumer.initialize());
    PROPAGATE_ERROR(disparityConsumer.waitRunning());

    for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++) {
        ISourceSettings *iSourceSettings = interface_cast<ISourceSettings>(iPreviewRequest[i]->getSourceSettings());
        iSourceSettings->setSensorMode(sensorModes[4]);
        iSourceSettings->setFrameDurationRange(Range<uint64_t>(1e9/FPS));
        EXIT_IF_NOT_OK(iPreviewRequest[i]->enableOutputStream(previewStream[i]),"Failed to enable stream in capture request");
        EXIT_IF_NOT_OK(iCaptureSession[i]->repeat(previewRequest[i]), "Unable to submit repeat() request");
    }

    //sleep(options.captureTime());
    sleep(1);

    for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++) {
        iCaptureSession[i]->stopRepeat();
        iCaptureSession[i]->waitForIdle();

        // Disconnect Argus producer from the EGLStreams (and unblock consumer acquire).
        PRODUCER_PRINT("Captures complete, disconnecting producer.\n");
        interface_cast<IEGLOutputStream>(previewStream[i])->disconnect();
    }
    
    PROPAGATE_ERROR(disparityConsumer.shutdown());
    PRODUCER_PRINT("Captures complete, shutdown thread.\n");
    for (uint32_t i = 0; i < MAX_CAM_DEVICE; i++) {
        if(captureSession[i])
        {
            captureSession[i]->destroy();
            captureSession[i] = NULL;
        }
        if(previewRequest[i])
        {
            previewRequest[i]->destroy();
            previewRequest[i] = NULL;
        }
        if(previewStream[i])
        {
            previewStream[i]->destroy();
            previewStream[i] = NULL;
        }
        if(streamSettings[i])
        {
            streamSettings[i]->destroy();
            streamSettings[i] = NULL;
        }
    }
    // Shut down Argus.
    cameraProvider.reset();
    PRODUCER_PRINT("Captures complete, reset .\n");
    // Cleanup the EGL display
    PROPAGATE_ERROR(g_display.cleanup());

    PRODUCER_PRINT("Done -- exiting.\n");
    return true;
}

}; // namespace ArgusSamples

int main(int argc, char *argv[])
{
    ArgusSamples::CommonOptions options(basename(argv[0]),
                                        ArgusSamples::CommonOptions::Option_T_CaptureTime);
    if (!options.parse(argc, argv))
        return EXIT_FAILURE;
    if (options.requestedExit())
        return EXIT_SUCCESS;

    if (!ArgusSamples::execute(options))
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
