package com.example.signdetect

import android.Manifest
import android.app.AlertDialog
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
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
import androidx.exifinterface.media.ExifInterface
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Detect Sign screen.
 *  - Live camera detection (front/back, switchable).
 *  - "Upload image" to detect a sign from a gallery photo; if no hand is found
 *    it shows: "Unable to detect — please upload a different picture".
 */
class DetectActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var predictionText: TextView
    private lateinit var sentenceText: TextView

    private lateinit var cameraExecutor: ExecutorService
    private var handLandmarker: HandLandmarker? = null
    private var classifier: HandClassifier? = null

    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private val confidenceThreshold = 0.55f

    // ---- Sentence building state ----
    // The sentence is committed one letter at a time: a letter is only added
    // when the same sign is held steady for HOLD_FRAMES consecutive frames, and
    // it won't repeat until the sign changes (or the hand disappears). Removing
    // the hand for NO_HAND_SPACE_FRAMES inserts a space, so words are separated.
    private val sentence = StringBuilder()
    private var candidateLabel: String? = null   // label currently being held
    private var candidateFrames = 0              // how long it's been held
    private var lastCommittedLabel: String? = null // last letter added (blocks repeats)
    private var noHandFrames = 0                 // frames since a hand was seen

    private val HOLD_FRAMES = 10                 // steady frames needed to commit a letter
    private val NO_HAND_SPACE_FRAMES = 15        // no-hand frames that insert a space
    private val SPACE_TOKEN = "SPACE"            // pseudo-label for the open-palm space gesture

    private val requestPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else predictionText.text = "Camera permission denied"
        }

    private val pickImage =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            if (uri != null) detectFromImage(uri)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_detect)

        previewView = findViewById(R.id.previewView)
        predictionText = findViewById(R.id.predictionText)
        sentenceText = findViewById(R.id.sentenceText)

        findViewById<Button>(R.id.uploadButton).setOnClickListener {
            pickImage.launch("image/*")
        }
        findViewById<Button>(R.id.switchCameraButton).setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT)
                CameraSelector.LENS_FACING_BACK else CameraSelector.LENS_FACING_FRONT
            startCamera()
        }
        findViewById<Button>(R.id.spaceButton).setOnClickListener { appendSpace() }
        findViewById<Button>(R.id.backspaceButton).setOnClickListener {
            if (sentence.isNotEmpty()) {
                sentence.deleteCharAt(sentence.length - 1)
                lastCommittedLabel = null
                sentenceText.text = sentence.toString()
            }
        }
        findViewById<Button>(R.id.clearButton).setOnClickListener {
            sentence.setLength(0)
            candidateLabel = null
            candidateFrames = 0
            lastCommittedLabel = null
            sentenceText.text = ""
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
        classifier = HandClassifier(this)
        handLandmarker = createLandmarker()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) startCamera() else requestPermission.launch(Manifest.permission.CAMERA)
    }

    private fun createLandmarker(): HandLandmarker {
        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build())
            .setRunningMode(RunningMode.IMAGE)
            .setNumHands(1)
            .setMinHandDetectionConfidence(0.5f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .build()
        return HandLandmarker.createFromOptions(this, options)
    }

    // ---------------- Live camera ----------------

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
        if (landmarker == null || clf == null) { imageProxy.close(); return }

        val bitmap = Bitmap.createBitmap(
            imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888
        )
        bitmap.copyPixelsFromBuffer(imageProxy.planes[0].buffer)

        val matrix = Matrix()
        matrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
        if (lensFacing == CameraSelector.LENS_FACING_FRONT) matrix.postScale(-1f, 1f)
        val upright = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

        val result = landmarker.detect(BitmapImageBuilder(upright).build())
        val hands = result.landmarks()
        if (hands.isNotEmpty()) {
            noHandFrames = 0
            if (HandGeometry.isOpenPalm(hands[0])) {
                // Dedicated space gesture — bypasses the A–Z classifier entirely.
                onStableSign(SPACE_TOKEN)
                runOnUiThread { predictionText.text = "␣ space" }
            } else {
                val (label, prob) = clf.predict(LandmarkFeatures.extract(hands[0]))
                if (prob >= confidenceThreshold) {
                    onStableSign(label)
                    runOnUiThread {
                        predictionText.text = "$label   ${(prob * 100).toInt()}%"
                    }
                } else {
                    // Low confidence — treat as "no clear sign", don't build the candidate.
                    candidateLabel = null
                    candidateFrames = 0
                    runOnUiThread { predictionText.text = "…" }
                }
            }
        } else {
            candidateLabel = null
            candidateFrames = 0
            noHandFrames++
            if (noHandFrames == NO_HAND_SPACE_FRAMES) {
                lastCommittedLabel = null // next sign can be committed again
                runOnUiThread { appendSpace() }
            }
            runOnUiThread { predictionText.text = "Show a hand sign…" }
        }
        imageProxy.close()
    }

    /**
     * Called every frame with a confident sign (an A–Z letter, or [SPACE_TOKEN]
     * for the open-palm space gesture). Commits it to the sentence once it has
     * been held steady, and blocks repeats until the sign changes. Runs on the
     * camera thread; only touches counters + posts UI work.
     */
    private fun onStableSign(key: String) {
        if (key == candidateLabel) {
            candidateFrames++
        } else {
            candidateLabel = key
            candidateFrames = 1
        }

        if (candidateFrames == HOLD_FRAMES && key != lastCommittedLabel) {
            lastCommittedLabel = key
            runOnUiThread {
                if (key == SPACE_TOKEN) {
                    if (sentence.isNotEmpty() && sentence.last() != ' ') sentence.append(' ')
                } else {
                    sentence.append(key)
                }
                sentenceText.text = sentence.toString()
            }
        }
    }

    /** Adds a single space to the sentence (ignored if the last char is already a space). */
    private fun appendSpace() {
        if (sentence.isNotEmpty() && sentence.last() != ' ') {
            sentence.append(' ')
            sentenceText.text = sentence.toString()
        }
    }

    // ---------------- Upload image ----------------

    private fun detectFromImage(uri: Uri) {
        val bitmap = decodeUprightBitmap(uri)
        if (bitmap == null) { showResult("Could not read that image. Try another."); return }

        // Use a short-lived IMAGE landmarker so we never clash with the camera thread.
        val landmarker = createLandmarker()
        val result = landmarker.detect(BitmapImageBuilder(bitmap).build())
        landmarker.close()

        val hands = result.landmarks()
        if (hands.isEmpty()) {
            showResult("Unable to detect — please upload a different picture.")
        } else {
            val clf = classifier
            if (clf == null) { showResult("Model not loaded."); return }
            val (label, prob) = clf.predict(LandmarkFeatures.extract(hands[0]))
            showResult("Detected sign:  $label\nConfidence:  ${(prob * 100).toInt()}%")
        }
    }

    private fun decodeUprightBitmap(uri: Uri): Bitmap? {
        return try {
            val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
            val bmp = contentResolver.openInputStream(uri).use {
                BitmapFactory.decodeStream(it, null, opts)
            } ?: return null

            // Respect EXIF rotation so the hand is upright for detection.
            val orientation = contentResolver.openInputStream(uri).use { input ->
                if (input == null) ExifInterface.ORIENTATION_NORMAL
                else ExifInterface(input).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
                )
            }
            val degrees = when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> 0f
            }
            if (degrees == 0f) bmp
            else Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height,
                Matrix().apply { postRotate(degrees) }, true)
        } catch (e: Exception) {
            null
        }
    }

    private fun showResult(message: String) {
        runOnUiThread {
            AlertDialog.Builder(this)
                .setTitle("Result")
                .setMessage(message)
                .setPositiveButton("OK", null)
                .show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        handLandmarker?.close()
    }
}
