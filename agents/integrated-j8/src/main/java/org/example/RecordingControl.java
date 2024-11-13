package org.example;



public class RecordingControl {
    private static boolean recording = false;

    public static void startRecording() {
        recording = true;
    }

    public static void stopRecording() {
        recording = false;
    }

    public static boolean isRecording() {
        return recording;
    }
}
