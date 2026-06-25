package com.example.signdetect

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.sqrt

/**
 * Turns one hand's 21 MediaPipe landmarks into the same 63-D translation/scale
 * invariant vector the Python `hand_utils.extract_features` produces:
 *   1. subtract the wrist (landmark 0) so position in the frame does not matter
 *   2. divide by the largest landmark distance so scale (distance to camera) does not matter
 */
object LandmarkFeatures {

    fun extract(landmarks: List<NormalizedLandmark>): FloatArray {
        val n = landmarks.size // expected 21
        val xs = FloatArray(n)
        val ys = FloatArray(n)
        val zs = FloatArray(n)

        val wx = landmarks[0].x()
        val wy = landmarks[0].y()
        val wz = landmarks[0].z()

        var maxDist = 0f
        for (i in 0 until n) {
            val x = landmarks[i].x() - wx
            val y = landmarks[i].y() - wy
            val z = landmarks[i].z() - wz
            xs[i] = x; ys[i] = y; zs[i] = z
            val d = sqrt(x * x + y * y + z * z)
            if (d > maxDist) maxDist = d
        }
        if (maxDist < 1e-6f) maxDist = 1f

        val out = FloatArray(n * 3)
        for (i in 0 until n) {
            out[i * 3] = xs[i] / maxDist
            out[i * 3 + 1] = ys[i] / maxDist
            out[i * 3 + 2] = zs[i] / maxDist
        }
        return out
    }
}
