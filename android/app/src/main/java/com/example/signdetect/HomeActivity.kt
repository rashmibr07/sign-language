package com.example.signdetect

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.activity.ComponentActivity

/** Landing screen: title + two buttons (Detect Sign, Record Data). */
class HomeActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_home)

        findViewById<Button>(R.id.detectButton).setOnClickListener {
            startActivity(Intent(this, DetectActivity::class.java))
        }
        findViewById<Button>(R.id.recordButton).setOnClickListener {
            startActivity(Intent(this, RecordActivity::class.java))
        }
    }
}
