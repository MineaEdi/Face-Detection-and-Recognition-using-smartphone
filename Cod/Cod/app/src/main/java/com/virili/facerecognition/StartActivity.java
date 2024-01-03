package com.virili.facerecognition;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;

public class StartActivity extends AppCompatActivity {

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_start);
    }
    public void startDetectionButtonClick(View view) {
        // Start the main activity
        Intent intent = new Intent(this, MainActivity.class); // Replace MainActivity with your actual main activity class
        startActivity(intent);
        finish(); // Optional: finish the current activity to prevent going back to it
    }

}
