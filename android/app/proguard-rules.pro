# Keep MediaPipe Tasks classes (they are accessed via JNI / reflection).
-keep class com.google.mediapipe.** { *; }
-keep class com.google.protobuf.** { *; }
-dontwarn com.google.mediapipe.**
