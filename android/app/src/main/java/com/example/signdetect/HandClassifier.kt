package com.example.signdetect

import android.content.Context
import org.json.JSONObject

/**
 * Loads the exported neural-network weights (classifier.json in assets) and runs
 * the exact same forward pass as the Python/scikit-learn model:
 *   hidden layers: z = a·W + b, then ReLU
 *   output layer:  z = a·W + b, then softmax
 *
 * Input is the 63-D normalised landmark vector produced by [LandmarkFeatures].
 * Output is the most likely label and its probability.
 */
class HandClassifier(context: Context) {

    private val labels: List<String>
    private val weights: List<Array<FloatArray>> // per layer: [inDim][outDim]
    private val biases: List<FloatArray>          // per layer: [outDim]

    init {
        val json = context.assets.open("classifier.json").bufferedReader().use { it.readText() }
        val root = JSONObject(json)

        val labelArr = root.getJSONArray("labels")
        labels = (0 until labelArr.length()).map { labelArr.getString(it) }

        val layerArr = root.getJSONArray("layers")
        val w = ArrayList<Array<FloatArray>>()
        val b = ArrayList<FloatArray>()
        for (l in 0 until layerArr.length()) {
            val layer = layerArr.getJSONObject(l)
            val wJson = layer.getJSONArray("W")          // [inDim][outDim]
            val inDim = wJson.length()
            val outDim = wJson.getJSONArray(0).length()
            val matrix = Array(inDim) { i ->
                val row = wJson.getJSONArray(i)
                FloatArray(outDim) { j -> row.getDouble(j).toFloat() }
            }
            val bJson = layer.getJSONArray("b")
            val bias = FloatArray(bJson.length()) { bJson.getDouble(it).toFloat() }
            w.add(matrix)
            b.add(bias)
        }
        weights = w
        biases = b
    }

    /** Returns (label, probability) for the best class. */
    fun predict(input: FloatArray): Pair<String, Float> {
        var a = input
        val lastLayer = weights.size - 1
        for (layer in weights.indices) {
            a = denseLayer(a, weights[layer], biases[layer])
            a = if (layer < lastLayer) relu(a) else softmax(a)
        }
        var bestIdx = 0
        for (i in a.indices) if (a[i] > a[bestIdx]) bestIdx = i
        return labels[bestIdx] to a[bestIdx]
    }

    private fun denseLayer(input: FloatArray, w: Array<FloatArray>, b: FloatArray): FloatArray {
        val out = FloatArray(b.size) { b[it] }
        for (i in input.indices) {
            val xi = input[i]
            if (xi == 0f) continue
            val row = w[i]
            for (j in out.indices) out[j] += xi * row[j]
        }
        return out
    }

    private fun relu(x: FloatArray): FloatArray {
        for (i in x.indices) if (x[i] < 0f) x[i] = 0f
        return x
    }

    private fun softmax(x: FloatArray): FloatArray {
        var max = x[0]
        for (v in x) if (v > max) max = v
        var sum = 0f
        for (i in x.indices) { x[i] = Math.exp((x[i] - max).toDouble()).toFloat(); sum += x[i] }
        for (i in x.indices) x[i] /= sum
        return x
    }
}
