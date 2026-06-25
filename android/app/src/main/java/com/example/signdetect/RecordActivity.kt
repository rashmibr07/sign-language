package com.example.signdetect

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Record Data screen.
 * Captures YOUR hand landmarks for a chosen letter and accumulates rows in the
 * same CSV format as the desktop dataset (label + 63 features). "Save & Share"
 * writes my_samples.csv and opens the share sheet so you can send it to your
 * computer, merge it into data/dataset.csv, retrain, and rebuild the APK.
 *
 * (On-device retraining isn't practical; recording + export is the workable path.)
 */
class RecordActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var labelInput: EditText
    private lateinit var statusText: TextView
    private lateinit var recordButton: Button

    private lateinit var cameraExecutor: ExecutorService
    private var handLandmarker: HandLandmarker? = null

    private val rows = ArrayList<String>()
    @Volatile private var recording = false
    private val lensFacing = CameraSelector.LENS_FACING_FRONT

    private val requestPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else statusText.text = "Camera permission denied"
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_record)

        previewView = findViewById(R.id.previewView)
        labelInput = findViewById(R.id.labelInput)
        statusText = findViewById(R.id.statusText)
        recordButton = findViewById(R.id.recordButton)

        recordButton.setOnClickListener {
            if (!recording && labelInput.text.toString().trim().isEmpty()) {
                Toast.makeText(this, "Type the letter first", Toast.LENGTH_SHORT).show()
            } else {
                recording = !recording
                recordButton.text = if (recording) "Pause" else "Start recording"
            }
        }
        findViewById<Button>(R.id.exportButton).setOnClickListener { exportCsv() }

        cameraExecutor = Executors.newSingleThreadExecutor()
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
            .build()
        return HandLandmarker.createFromOptions(this, options)
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
                statusText.text = "Camera error: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(imageProxy: ImageProxy) {
        val landmarker = handLandmarker
        if (landmarker == null) { imageProxy.close(); return }

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

        if (recording && hands.isNotEmpty()) {
            val label = labelInput.text.toString().trim()
            val feats = LandmarkFeatures.extract(hands[0])
            val row = StringBuilder(label)
            for (v in feats) { row.append(','); row.append(v) }
            synchronized(rows) { rows.add(row.toString()) }
            runOnUiThread { statusText.text = "Collected: ${rows.size} rows" }
        } else if (hands.isEmpty() && recording) {
            runOnUiThread { statusText.text = "No hand — collected: ${rows.size} rows" }
        }
        imageProxy.close()
    }

    private fun exportCsv() {
        val snapshot: List<String>
        synchronized(rows) { snapshot = ArrayList(rows) }
        if (snapshot.isEmpty()) {
            Toast.makeText(this, "Nothing recorded yet", Toast.LENGTH_SHORT).show()
            return
        }
        try {
            val dir = getExternalFilesDir(null)
            val file = File(dir, "my_samples.csv")
            file.bufferedWriter().use { w ->
                w.append("label")
                for (i in 0 until 63) { w.append(",f"); w.append(i.toString()) }
                w.append('\n')
                for (r in snapshot) { w.append(r); w.append('\n') }
            }
            val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
            val share = Intent(Intent.ACTION_SEND).apply {
                type = "text/csv"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(share, "Share my_samples.csv (${snapshot.size} rows)"))
        } catch (e: Exception) {
            Toast.makeText(this, "Export failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        handLandmarker?.close()
    }
}
