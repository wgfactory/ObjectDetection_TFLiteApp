/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.wgf.objectapp_tflite.main

import android.graphics.*
import android.media.ImageReader
import android.os.SystemClock
import android.util.Size
import android.util.TypedValue
import android.view.View
import android.widget.Toast
import com.wgf.objectapp_tflite.R
import com.wgf.objectapp_tflite.detection.OverlayView
import com.wgf.objectapp_tflite.env.BorderedText
import com.wgf.objectapp_tflite.env.ImageUtils
import com.wgf.objectapp_tflite.env.Logger
import com.wgf.objectapp_tflite.traking.MultiBoxTracker
import org.tensorflow.lite.examples.detection.tflite.Detector
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel
import java.io.IOException
import java.util.*

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
class DetectorActivity : CameraActivity(),
    ImageReader.OnImageAvailableListener {
    var trackingOverlay: OverlayView? = null
    private var sensorOrientation: Int? = null
    private var detector: Detector? = null
    private var lastProcessingTimeMs: Long = 0
    private var rgbFrameBitmap: Bitmap? = null
    private var croppedBitmap: Bitmap? = null
    private var cropCopyBitmap: Bitmap? = null
    private var computingDetection = false
    private var timestamp: Long = 0
    private var frameToCropTransform: Matrix? = null
    private var cropToFrameTransform: Matrix? = null
    private var tracker: MultiBoxTracker? = null
    private var borderedText: BorderedText? = null
    public override fun onPreviewSizeChosen(size: Size, rotation: Int) {
        val textSizePx = TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            TEXT_SIZE_DIP,
            resources.displayMetrics
        )
        borderedText = BorderedText(textSizePx)
        borderedText!!.setTypeface(Typeface.MONOSPACE)
        tracker = MultiBoxTracker(this)
        var cropSize = TF_OD_API_INPUT_SIZE
        try {
            detector =
                TFLiteObjectDetectionAPIModel.create(
                    this,
                    TF_OD_API_MODEL_FILE,
                    TF_OD_API_LABELS_FILE,
                    TF_OD_API_INPUT_SIZE,
                    TF_OD_API_IS_QUANTIZED
                )
            cropSize = TF_OD_API_INPUT_SIZE
        } catch (e: IOException) {
            e.printStackTrace()
            LOGGER.e(e, "Exception initializing Detector!")
            val toast = Toast.makeText(
                applicationContext,
                "Detector could not be initialized",
                Toast.LENGTH_SHORT
            )
            toast.show()
            finish()
        }
        previewWidth = size.width
        previewHeight = size.height
        sensorOrientation = rotation - screenOrientation
        LOGGER.i(
            "Camera orientation relative to screen canvas: %d",
            sensorOrientation
        )
        LOGGER.i(
            "Initializing at size %dx%d",
            previewWidth,
            previewHeight
        )
        rgbFrameBitmap = Bitmap.createBitmap(
            previewWidth,
            previewHeight,
            Bitmap.Config.ARGB_8888
        )
        croppedBitmap = Bitmap.createBitmap(
            cropSize,
            cropSize,
            Bitmap.Config.ARGB_8888
        )
        frameToCropTransform = ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation!!, MAINTAIN_ASPECT
        )
        cropToFrameTransform = Matrix()
        frameToCropTransform!!.invert(cropToFrameTransform)
        trackingOverlay =
            findViewById<View>(R.id.tracking_overlay) as OverlayView
        trackingOverlay!!.addCallback { canvas ->
            tracker!!.draw(canvas)
            if (isDebug) {
                tracker!!.drawDebug(canvas)
            }
        }
        tracker!!.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation!!)
    }

    override fun processImage() {
        ++timestamp
        val currTimestamp = timestamp
        trackingOverlay!!.postInvalidate()

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage()
            return
        }
        computingDetection = true
        LOGGER.i("Preparing image $currTimestamp for detection in bg thread.")
        rgbFrameBitmap!!.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight)
        readyForNextImage()
        val canvas = Canvas(croppedBitmap!!)
        canvas.drawBitmap(rgbFrameBitmap!!, frameToCropTransform!!, null)
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap)
        }
        runInBackground {
            LOGGER.i("Running detection on image $currTimestamp")
            val startTime = SystemClock.uptimeMillis()
            val results =
                detector!!.recognizeImage(croppedBitmap)
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime
            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap!!)
            val canvas = Canvas(cropCopyBitmap!!)
            val paint = Paint()
            paint.color = Color.RED
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 2.0f
            var minimumConfidence =
                MINIMUM_CONFIDENCE_TF_OD_API
            minimumConfidence = when (MODE) {
                DetectorMode.TF_OD_API -> MINIMUM_CONFIDENCE_TF_OD_API
            }
            val mappedRecognitions: MutableList<Detector.Recognition> =
                ArrayList()
            for (result in results) {
                val location = result.location
                if (location != null && result.confidence >= minimumConfidence) {
                    canvas.drawRect(location, paint)
                    cropToFrameTransform!!.mapRect(location)
                    result.location = location
                    mappedRecognitions.add(result)
                }
            }
            tracker!!.trackResults(mappedRecognitions, currTimestamp)
            trackingOverlay!!.postInvalidate()
            computingDetection = false
            runOnUiThread { /*showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");*/
            }
        }
    }

    override fun getLayoutId(): Int {
        return R.layout.fragment_camera2_basic
    }

    override fun getDesiredPreviewFrameSize(): Size {
        return DESIRED_PREVIEW_SIZE
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum class DetectorMode {
        TF_OD_API
    }

    override fun setUseNNAPI(isChecked: Boolean) {
        runInBackground {
            try {
                detector!!.setUseNNAPI(isChecked)
            } catch (e: UnsupportedOperationException) {
                LOGGER.e(e, "Failed to set \"Use NNAPI\".")
                runOnUiThread {
                    Toast.makeText(
                        this,
                        e.message,
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }

    override fun setNumThreads(numThreads: Int) {
        runInBackground {
            try {
                detector!!.setNumThreads(numThreads)
            } catch (e: IllegalArgumentException) {
                LOGGER.e(e, "Failed to set multithreads.")
                runOnUiThread {
                    Toast.makeText(
                        this,
                        e.message,
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }

    companion object {
        private val LOGGER =
            Logger()

        // Configuration values for the prepackaged SSD model.
        private const val TF_OD_API_INPUT_SIZE = 300
        private const val TF_OD_API_IS_QUANTIZED = true
        private const val TF_OD_API_MODEL_FILE = "detect.tflite"
        private const val TF_OD_API_LABELS_FILE = "labelmap.txt"
        private val MODE = DetectorMode.TF_OD_API

        // Minimum detection confidence to track a detection.
        private const val MINIMUM_CONFIDENCE_TF_OD_API = 0.5f
        private const val MAINTAIN_ASPECT = false
        private val DESIRED_PREVIEW_SIZE = Size(640, 480)
        private const val SAVE_PREVIEW_BITMAP = false
        private const val TEXT_SIZE_DIP = 10f
    }
}