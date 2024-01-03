package com.virili.facerecognition;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;
import android.graphics.RectF;

import androidx.annotation.Nullable;

public class GraphicOverlay extends View {
    private RectF boundingBox;
    private String recognizedName;
    private String confidence;

    public GraphicOverlay(Context context) {
        super(context);
    }

    public GraphicOverlay(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
    }

    public GraphicOverlay(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public GraphicOverlay(Context context, @Nullable AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
    }

    public void updateOverlay(RectF boundingBox, String recognizedName, String confidence) {
        this.boundingBox = boundingBox;
        this.recognizedName = recognizedName;
        this.confidence = confidence;
        postInvalidate(); // Trigger redraw
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if (boundingBox != null) {
            // Draw a border around the detected face
            Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(5.0f);
            canvas.drawRect(boundingBox, paint);

            // Draw the recognized face's name and confidence level above the box
            if (recognizedName != null && !recognizedName.isEmpty()) {
                paint.setColor(Color.WHITE);
                paint.setStyle(Paint.Style.FILL);
                paint.setTextSize(60.0f);

                // Draw recognized name
                float textWidthName = paint.measureText(recognizedName);
                float xName = boundingBox.centerX() - (textWidthName / 2);
                float yName = boundingBox.top - 10; // Adjust the distance above the box
                canvas.drawText(recognizedName, xName, yName, paint);

                // Draw confidence level
                if (confidence != null && !confidence.isEmpty()) {
                    paint.setColor(Color.GREEN); // Choose the color for confidence level
                    paint.setTextSize(50.0f); // Choose the text size for confidence level
                    float textWidthConfidence = paint.measureText(confidence);
                    float xConfidence = boundingBox.centerX() - (textWidthConfidence / 2);
                    float yConfidence = yName + 40; // Adjust the distance below the recognized name
                    canvas.drawText(confidence, xConfidence, yConfidence, paint);
                }
            }

        }
    }
    //help me do a clear overlay
    public void clearOverlay() {
        boundingBox = null;
        recognizedName = null;
        confidence = null;
        postInvalidate(); // Trigger redraw
    }
}