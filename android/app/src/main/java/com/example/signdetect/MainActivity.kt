package com.example.signdetect

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.widget.Button
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var predictionText: TextView

    private lateinit var cameraExecutor: ExecutorService
    private var handLandmarker: HandLandmarker? = null
    private var classifier: HandClassifier? = null

    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private val confidenceThreshold = 0.55f

    private val requestPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else predictionText.text = "Camera permission denied"
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        predictionText = findViewById(R.id.predictionText)
        findViewById<Button>(R.id.switchCameraButton).setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT)
                CameraSelector.LENS_FACING_BACK else CameraSelector.LENS_FACING_FRONT
            startCamera()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
        classifier = HandClassifier(this)
        setupHandLandmarker()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun setupHandLandmarker() {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .build()
        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.VIDEO)
            .setNumHands(1)
            .setMinHandDetectionConfidence(0.5f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .build()
        handLandmarker = HandLandmarker.createFromOptions(this, options)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(cameraExecutor, ::analyze) }

            val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, selector, preview, analysis)
            } catch (e: Exception) {
                predictionText.text = "Camera error: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(imageProxy: ImageProxy) {
        val landmarker = handLandmarker
        val clf = classifier
        if (landmarker == null || clf == null) {
            imageProxy.close(); return
        }

        // CameraX RGBA_8888 frame -> Bitmap
        val bitmap = Bitmap.createBitmap(
            imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888
        )
        bitmap.copyPixelsFromBuffer(imageProxy.planes[0].buffer)

        // Rotate to upright, and mirror for the front camera (matches training data).
        val matrix = Matrix()
        matrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
        if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
            matrix.postScale(-1f, 1f)
        }
        val upright = Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
        )

        val mpImage = BitmapImageBuilder(upright).build()
        val result = landmarker.detectForVideo(mpImage, SystemClock.uptimeMillis())

        val hands = result.landmarks()
        if (hands.isNotEmpty()) {
            val features = LandmarkFeatures.extract(hands[0])
            val (label, prob) = clf.predict(features)
            runOnUiThread {
                predictionText.text =
                    if (prob >= confidenceThreshold) "$label   ${(prob * 100).toInt()}%"
                    else "…"
            }
        } else {
            runOnUiThread { predictionText.text = "Show a hand sign…" }
        }

        imageProxy.close()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        handLandmarker?.close()
    }
}
