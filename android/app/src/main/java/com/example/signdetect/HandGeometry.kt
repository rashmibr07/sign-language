package com.example.signdetect

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.sqrt

/**
 * Lightweight geometric checks on the raw 21 MediaPipe hand landmarks, used for
 * gestures the A–Z classifier cannot represent (e.g. an open flat palm = space).
 *
 * Landmark indices (MediaPipe hand model):
 *   0 wrist
 *   thumb  1 CMC  2 MCP  3 IP   4 TIP
 *   index  5 MCP  6 PIP  7 DIP  8 TIP
 *   middle 9 MCP 10 PIP 11 DIP 12 TIP
 *   ring  13 MCP 14 PIP 15 DIP 16 TIP
 *   pinky 17 MCP 18 PIP 19 DIP 20 TIP
 */
object HandGeometry {

    private fun dist(a: NormalizedLandmark, b: NormalizedLandmark): Float {
        val dx = a.x() - b.x()
        val dy = a.y() - b.y()
        val dz = a.z() - b.z()
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    /**
     * True for an open, spread flat palm (all five fingers extended and fanned
     * out). This is distinct from letter "B" (fingers extended but held together)
     * because of the fingertip-spread requirement.
     */
    fun isOpenPalm(lm: List<NormalizedLandmark>): Boolean {
        if (lm.size < 21) return false
        val wrist = lm[0]

        // Each finger is "extended" when its tip is clearly farther from the wrist
        // than its PIP joint — orientation-independent, unlike a simple y-compare.
        val fingers = arrayOf(intArrayOf(8, 6), intArrayOf(12, 10), intArrayOf(16, 14), intArrayOf(20, 18))
        for (f in fingers) {
            if (dist(lm[f[0]], wrist) <= dist(lm[f[1]], wrist) * 1.1f) return false
        }
        // Thumb extended (tip vs MCP).
        if (dist(lm[4], wrist) <= dist(lm[2], wrist) * 1.1f) return false

        // Fingers must be spread: average gap between adjacent fingertips, measured
        // against palm size (wrist → middle-finger MCP), must exceed a threshold.
        val scale = dist(lm[0], lm[9]).coerceAtLeast(1e-6f)
        val tips = intArrayOf(4, 8, 12, 16, 20)
        var gap = 0f
        for (i in 0 until tips.size - 1) gap += dist(lm[tips[i]], lm[tips[i + 1]])
        val avgGap = gap / (tips.size - 1)
        return avgGap > 0.55f * scale
    }
}
