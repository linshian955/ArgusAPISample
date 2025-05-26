/*
 * Copyright (c) 2016 - 2017, NVIDIA CORPORATION. All rights reserved.
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

#include "CaptureConsumer.h"
#include "Error.h"

#include <Argus/Argus.h>
#include <string.h>
#include <stdlib.h>
#include <sstream>
#include <iomanip>
#include <Argus/Ext/SensorTimestampTsc.h>

namespace ArgusSamples
{

#define JPEG_CONSUMER_PRINT(...)    printf("JPEG CONSUMER: " __VA_ARGS__)

#ifdef ANDROID
#define JPEG_PREFIX "/sdcard/DCIM/Argus_"
#else
#define JPEG_PREFIX "Argus_"
#endif

#define NENOTOMILLI 1000000

bool CaptureConsumerThread::threadInitialize()
{
    // Create the FrameConsumer.
    m_consumer = UniqueObj<FrameConsumer>(FrameConsumer::create(m_stream));
    if (!m_consumer)
        ORIGINATE_ERROR("Failed to create FrameConsumer");

    return true;
}

bool CaptureConsumerThread::threadExecute()
{

    IEGLOutputStream *iEGLOutputStream = interface_cast<IEGLOutputStream>(m_stream);
    IFrameConsumer *iFrameConsumer = interface_cast<IFrameConsumer>(m_consumer);

    // Wait until the producer has connected to the stream.
    JPEG_CONSUMER_PRINT("Waiting until producer is connected...\n");
    if (iEGLOutputStream->waitUntilConnected() != STATUS_OK)
        ORIGINATE_ERROR("Stream failed to connect.");
    JPEG_CONSUMER_PRINT("Producer has connected; continuing.\n");;
    const uint64_t FIVE_SECONDS = 5000000000;
    Argus::Status status;
    std::ostringstream fileName;
    unsigned long long tscTimeStampNew = 0;
    unsigned long long frameNumber = 0;
    uint64_t frameduration = 0;
    Ext::ISensorTimestampTsc *iSensorTimestampTsc = NULL;
    std::string fileType = imageType == 0 ? ".jpg": ".yuv";
    while (true)
    {
        // Acquire a Frame.
        UniqueObj<Frame> frame(iFrameConsumer->acquireFrame(FIVE_SECONDS, &status));
        IFrame *iFrame = interface_cast<IFrame>(frame);
        if (!iFrame)
            break;

        //generate file name
        CaptureMetadata* captureMetadata =
                interface_cast<IArgusCaptureMetadata>(frame)->getMetadata();
        ICaptureMetadata* iMetadata = interface_cast<ICaptureMetadata>(captureMetadata);
        
        frameNumber = iFrame->getNumber();
        frameduration = iMetadata->getFrameDuration();
        iSensorTimestampTsc = interface_cast<Ext::ISensorTimestampTsc>(captureMetadata);
        if (iSensorTimestampTsc)
        {
            tscTimeStampNew = iSensorTimestampTsc->getSensorSofTimestampTsc();
        }
        
        fileName << JPEG_PREFIX <<"[T"<<tscTimeStampNew/NENOTOMILLI<<"][FNum_"<<frameNumber<<"][Cam_"<<cameraId<<"][D"<< frameduration <<"]"<< fileType;
        JPEG_CONSUMER_PRINT("frameNumberRight %lld cameraId %d frameduration %ld tscTimeStampRightNew %lld .\n",
            frameNumber,
            cameraId,
            frameduration,
            tscTimeStampNew);


        // Get the Frame's Image.
        Image *image = iFrame->getImage();
        if(imageType == 0)
        {
            IImageJPEG *iJPEG = interface_cast<IImageJPEG>(image);
        
            if (!iJPEG)
                ORIGINATE_ERROR("Failed to get IImageJPEG interface.");

            if (iJPEG->writeJPEG(fileName.str().c_str()) == STATUS_OK)
                JPEG_CONSUMER_PRINT("Captured a still image to '%s'\n", fileName.str().c_str());
            else
                ORIGINATE_ERROR("Failed to write JPEG to '%s'\n", fileName.str().c_str());
        }
        else
        {
            EGLStream::IImage *yuvIImage = Argus::interface_cast<EGLStream::IImage>(image);
            if(!yuvIImage)
                ORIGINATE_ERROR("Failed to get YUV IImage");

            EGLStream::IImage2D *yuvIImage2D = Argus::interface_cast<EGLStream::IImage2D>(image);
            if(!yuvIImage2D)
                ORIGINATE_ERROR("Failed to get YUV iImage2D");

            EGLStream::IImageHeaderlessFile *yuvIImageHeaderlessFile =
            Argus::interface_cast<EGLStream::IImageHeaderlessFile>(image);
            if(!yuvIImageHeaderlessFile)
                ORIGINATE_ERROR("Failed to get YUV IImageHeaderlessFile");

            status = yuvIImageHeaderlessFile->writeHeaderlessFile(fileName.str().c_str());
            printf("Wrote YUV file : %s\n", fileName.str().c_str());
           // Wait for CAPTURE_TIME seconds.
       }

        
    }

    JPEG_CONSUMER_PRINT("No more frames. Cleaning up.\n");

    PROPAGATE_ERROR(requestShutdown());

    return true;
}

bool CaptureConsumerThread::threadShutdown()
{
    JPEG_CONSUMER_PRINT("Done.\n");

    return true;
}

} // namespace ArgusSamples
