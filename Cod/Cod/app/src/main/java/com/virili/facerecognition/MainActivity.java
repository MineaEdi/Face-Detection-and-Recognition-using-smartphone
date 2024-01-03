package com.virili.facerecognition;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.media.Image;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;

import androidx.annotation.NonNull;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AlertDialog;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.common.util.concurrent.ListenableFuture;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import android.os.Environment;
import android.os.ParcelFileDescriptor;
import android.text.InputType;
import android.util.Pair;
import android.util.Size;
import android.view.View;

import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;


public class MainActivity extends AppCompatActivity {
    FaceDetector detector;

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    PreviewView previewView;
    ImageView face_preview;

    GraphicOverlay graphicOverlay;
    Interpreter tfLite;
    TextView reco_name, preview_info;
    Button recognize, camera_switch, menu;
    ImageButton add_face;
    CameraSelector cameraSelector;
    boolean start = true, flipX = false;
    boolean showDetected = false;
    Context context = MainActivity.this;
    int cam_face = CameraSelector.LENS_FACING_BACK; //Default Back Camera
    ProcessCameraProvider cameraProvider;
    private static int SELECT_PICTURE = 1;
    private static final int MY_CAMERA_REQUEST_CODE = 100;
    String modelFile = "mobile_face_net.tflite"; //model name
    int inputSize = 112;  //Input size for model
    float[][] embeedings;
    float IMAGE_MEAN = 128.0f;
    float IMAGE_STD = 128.0f;
    int OUTPUT_SIZE = 192; //Output size of model
    float distance = 1.0f; //Threshold distance between 2 faces

    boolean isRecognizing = false;
    Bitmap scaled;
    private HashMap<String, ImageClassification.Recognition> registered = new HashMap<>(); //saved Faces for recognition
    private HashMap<String, Bitmap> savedFaces = new HashMap<>(); //saved Faces from detected faces

    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        face_preview = findViewById(R.id.imageView);

        reco_name = findViewById(R.id.textView);

        preview_info = findViewById(R.id.textView2);

        add_face = findViewById(R.id.imageButton);
        add_face.setVisibility(View.INVISIBLE);
        face_preview.setVisibility(View.INVISIBLE);
        recognize = findViewById(R.id.button3);
        camera_switch = findViewById(R.id.button5);
        menu = findViewById(R.id.button);

        graphicOverlay = findViewById(R.id.graphic_overlay);


        //Camera Permission
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, MY_CAMERA_REQUEST_CODE);
        }

        LoadModel();

        //On-screen switch to toggle between Cameras.
        camera_switch.setOnClickListener(v -> {
            if (cam_face == CameraSelector.LENS_FACING_BACK) {
                cam_face = CameraSelector.LENS_FACING_FRONT;
                flipX = true;
            } else {
                cam_face = CameraSelector.LENS_FACING_BACK;
                flipX = false;
            }
            cameraProvider.unbindAll();
            cameraBind();
        });


        menu.setOnClickListener(v -> {
            AlertDialog.Builder builder = new AlertDialog.Builder(context);
            builder.setTitle("Select Action:");

            // add a checkbox list
            String[] names = {"Save image on disk", "Load image from disk"};

            builder.setItems(names, (dialog, which) -> {

                switch (which) {
                    case 0:
                        saveFacesToGallery();
                        break;
                    case 1:
                        loadFacesFromGallery();
                        break;
                }

            });
            builder.setPositiveButton("OK", (dialog, which) -> {

            });
            builder.setNegativeButton("Cancel", null);

            // create and show the alert dialog
            AlertDialog dialog = builder.create();
            dialog.show();
        });

        add_face.setOnClickListener((v -> addFace()));

        recognize.setOnClickListener(v -> {
            if (recognize.getText().toString().equals("Recognize")) {
                isRecognizing = true;
                start = true;
                showDetected = false;
                recognize.setText("Detect Face");
                add_face.setVisibility(View.INVISIBLE);
                reco_name.setVisibility(View.VISIBLE);
                face_preview.setVisibility(View.INVISIBLE);
                preview_info.setText("");

            } else {
                isRecognizing = false;
                showDetected = true;
                recognize.setText("Recognize");
                add_face.setVisibility(View.VISIBLE);
                reco_name.setVisibility(View.INVISIBLE);
                face_preview.setVisibility(View.VISIBLE);
                preview_info.setText("No face detected yet.\n");

            }

        });


        cameraBind();

    }

    private void saveFacesToGallery() {

        if (!savedFaces.isEmpty()) {
            for (Map.Entry<String, Bitmap> entry : savedFaces.entrySet()) {
                String name = entry.getKey();
                Bitmap bitmap = entry.getValue();

                saveBitmapToFile(name, bitmap);
            }
        } else {

        }
    }


    private void addFace() {
        start = false;

        AlertDialog.Builder builder = new AlertDialog.Builder(context);
        builder.setTitle("Introduceti numele");

        // Set up the input
        final EditText input = new EditText(context);
        input.setInputType(InputType.TYPE_CLASS_TEXT);
        builder.setView(input);

        // Set up the buttons
        builder.setPositiveButton("ADD", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                ImageClassification.Recognition result = new ImageClassification.Recognition(
                        "0", "", -1f);
                result.setExtra(embeedings);
                savedFaces.put(input.getText().toString(), scaled);
                registered.put( input.getText().toString(),result);
                start=true;
            }
        });

        builder.setNegativeButton("Cancel", (dialog, which) -> {
            start = true;
            dialog.cancel();
        });
        builder.show();
    }

    void LoadModel() {
        try {
            tfLite = new Interpreter(loadModelFile(MainActivity.this, modelFile));
        } catch (IOException e) {
            e.printStackTrace();
        }
        //Initialize Face Detector
        FaceDetectorOptions highAccuracyOpts =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                        .build();
        detector = FaceDetection.getClient(highAccuracyOpts);
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == MY_CAMERA_REQUEST_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "camera permission granted", Toast.LENGTH_LONG).show();
            } else {
                Toast.makeText(this, "camera permission denied", Toast.LENGTH_LONG).show();
            }
        }
    }

    private MappedByteBuffer loadModelFile(Activity activity, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    //Bind camera and preview view
    private void cameraBind() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        previewView = findViewById(R.id.previewView);
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();

                bindPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {

            }
        }, ContextCompat.getMainExecutor(this));
    }

    void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder()
                .build();

        cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(cam_face)
                .build();

        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        ImageAnalysis imageAnalysis =
                new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST) //Latest frame is shown
                        .build();

        Executor executor = Executors.newSingleThreadExecutor();
        imageAnalysis.setAnalyzer(executor, imageProxy -> {
            try {
                Thread.sleep(0);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            InputImage image = null;

            @SuppressLint("UnsafeExperimentalUsageError")

            Image mediaImage = imageProxy.getImage();

            if (mediaImage != null) {
                image = InputImage.fromMediaImage(mediaImage, imageProxy.getImageInfo().getRotationDegrees());
            }


            //Process acquired image to detect faces
            @SuppressLint("SuspiciousIndentation") Task<List<Face>> result =
                    detector.process(image)
                            .addOnSuccessListener(
                                    faces -> {

                                        if (faces.size() != 0) {
                                            graphicOverlay.clearOverlay();
                                            Face face = faces.get(0); //Get first face from detected faces

                                            //mediaImage to Bitmap
                                            Bitmap frame_bmp = toBitmap(mediaImage);

                                            int rot = imageProxy.getImageInfo().getRotationDegrees();

                                            //Adjust orientation of Face ( e rotită imaginea la 90 de grade idk why)
                                            Bitmap frame_bmp1 = rotateBitmap(frame_bmp, rot, false, false);

                                            //Get bounding box of face
                                            RectF boundingBox = new RectF(face.getBoundingBox());

                                            //Crop out bounding box from whole Bitmap(image)
                                            Bitmap cropped_face = getCropBitmap(frame_bmp1, boundingBox);

                                            if (flipX)
                                                cropped_face = rotateBitmap(cropped_face, 0, flipX, false);
                                            //Scale the acquired Face to 112*112 which is required input for model
                                            scaled = getResizedBitmap(cropped_face, 112, 112);

                                            if (start) {
                                                face_preview.setImageBitmap(scaled);
                                                recognizeImage(scaled,boundingBox);
                                            }

                                        } else {
                                            graphicOverlay.clearOverlay();
                                            if (registered.isEmpty())
                                                reco_name.setText("No Face Added Yet");
                                            else
                                                reco_name.setText("You added some faces, please put a saved face in the frame");
                                        }

                                    })
                            .addOnFailureListener(
                                    e -> {

                                    })
                            .addOnCompleteListener(task -> {

                                imageProxy.close(); //v.important to acquire next frame for analysis
                            });
        });
        cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, imageAnalysis, preview);
    }

    public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        // CREATE A MATRIX FOR THE MANIPULATION
        Matrix matrix = new Matrix();
        // RESIZE THE BIT MAP
        matrix.postScale(scaleWidth, scaleHeight);

        // "RECREATE" THE NEW BITMAP
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, false);
        bm.recycle();
        return resizedBitmap;
    }

    private static Bitmap getCropBitmap(Bitmap image, RectF cropRectF) {
        Bitmap resultBitmap = Bitmap.createBitmap((int) cropRectF.width(),
                (int) cropRectF.height(), Bitmap.Config.ARGB_8888);
        Canvas cavas = new Canvas(resultBitmap);

        // draw background
        Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
        paint.setColor(Color.WHITE);
        cavas.drawRect(
                new RectF(0, 0, cropRectF.width(), cropRectF.height()),
                paint);


        Matrix matrix = new Matrix();
        matrix.postTranslate(-cropRectF.left, -cropRectF.top);

        cavas.drawBitmap(image, matrix, paint);

        if (image != null && !image.isRecycled()) {
            image.recycle();
        }

        return resultBitmap;
    }

    private static Bitmap rotateBitmap(
            Bitmap bitmap, int rotationDegrees, boolean flipX, boolean flipY) {
        Matrix matrix = new Matrix();

        // Rotate the image back to straight.
        matrix.postRotate(rotationDegrees);

        // Mirror the image along the X or Y axis.
        matrix.postScale(flipX ? -1.0f : 1.0f, flipY ? -1.0f : 1.0f);
        Bitmap rotatedBitmap =
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        // Recycle the old bitmap if it has changed.
        if (rotatedBitmap != bitmap) {
            bitmap.recycle();
        }
        return rotatedBitmap;
    }

    //IMPORTANT. If conversion not done ,the toBitmap conversion does not work on some devices.
    private static byte[] YUV_420_888toNV21(Image image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int ySize = width * height;
        int uvSize = width * height / 4;

        byte[] nv21 = new byte[ySize + uvSize * 2];

        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer(); // Y
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer(); // U
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer(); // V

        int rowStrideUV = image.getPlanes()[1].getRowStride();
        int pixelStrideUV = image.getPlanes()[1].getPixelStride();

        yBuffer.get(nv21, 0, ySize);

        for (int row = 0; row < height / 2; row++) {
            for (int col = 0; col < width / 2; col++) {
                int uvPos = col * pixelStrideUV + row * rowStrideUV;
                nv21[ySize++] = vBuffer.get(uvPos);
                nv21[ySize++] = uBuffer.get(uvPos);
            }
        }

        return nv21;
    }

    private Bitmap toBitmap(Image image) {
        try {
            byte[] nv21 = YUV_420_888toNV21(image);

            YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);

            ByteArrayOutputStream out = new ByteArrayOutputStream();
            yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

            byte[] imageBytes = out.toByteArray();

            return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
        } catch (Exception e) {
            e.printStackTrace();

            return null;
        }
    }


    private void loadFacesFromGallery() {
        start = false;
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), SELECT_PICTURE);
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == SELECT_PICTURE && data != null) {
            Uri selectedImageUri = data.getData();
            try {
                Bitmap frameBitmap = getBitmapFromUri(selectedImageUri);
                processImage(frameBitmap);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }


    private void processImage(Bitmap frameBitmap) {
        InputImage inputImage = InputImage.fromBitmap(frameBitmap, 0);

        detector.process(inputImage)
                .addOnSuccessListener(faces -> {
                    if (!faces.isEmpty()) {
                        graphicOverlay.clearOverlay();
                        Face face = faces.get(0);
                        Bitmap croppedFace = getCropBitmap(rotateBitmap(frameBitmap, 0, flipX, false), new RectF(face.getBoundingBox()));
                        scaled = getResizedBitmap(croppedFace, 112, 112);
                        face_preview.setImageBitmap(scaled);
                        RectF boundingBox = new RectF(face.getBoundingBox());
                        recognizeImage(scaled, boundingBox);
                        addFace();

                    }
                })
                .addOnFailureListener(e -> {
                    start = true;
                    Toast.makeText(context, "Failed to add", Toast.LENGTH_SHORT).show();
                });


    }

    //TAKE DATA FROM LOCAL STORAGE AND CONVERT TO BITMAP
    private Bitmap getBitmapFromUri(Uri uri) throws IOException {
        ParcelFileDescriptor parcelFileDescriptor =
                getContentResolver().openFileDescriptor(uri, "r");
        FileDescriptor fileDescriptor = parcelFileDescriptor.getFileDescriptor();
        Bitmap image = BitmapFactory.decodeFileDescriptor(fileDescriptor);
        parcelFileDescriptor.close();
        return image;
    }

    ////SAVING IMAGE TO FILE ON DEVICE
    private void saveBitmapToFile(String fileName, Bitmap bitmap) {
        // Get the directory where you want to save the images (e.g., external storage)
        File directory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);

        // Create a new file with the specified name
        File file = new File(directory, fileName + ".png");

        try {
            // Create a FileOutputStream to write the bitmap to the file
            FileOutputStream fos = new FileOutputStream(file);

            // Compress the bitmap to PNG format (you can choose other formats as well)
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);

            // Flush and close the stream
            fos.flush();
            fos.close();

            //added the saved file to the system's media store
            MediaScannerConnection.scanFile(
                    this,
                    new String[]{file.toString()},
                    null,
                    (path, uri) -> {
                        // Handle the scan completion if needed
                    }
            );


            Toast.makeText(this, "Saved: " + file.getAbsolutePath(), Toast.LENGTH_SHORT).show();
        } catch (IOException e) {

            e.printStackTrace();
        }
        savedFaces.clear();
    }

    public void recognizeImage(final Bitmap bitmap, RectF boundingBox) {

        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(inputSize, inputSize, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(IMAGE_MEAN, IMAGE_STD))
                .build();

        TensorImage inputImage = new TensorImage(DataType.FLOAT32);
        inputImage.load(bitmap);
        inputImage = imageProcessor.process(inputImage);

        // imgData is input to our model
        ByteBuffer imgData = inputImage.getBuffer();

        // Run model
        Object[] inputArray = {imgData};
        Map<Integer, Object> outputMap = new HashMap<>();
        embeedings = new float[1][OUTPUT_SIZE];
        outputMap.put(0, embeedings);
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

        float distance_local = Float.MAX_VALUE;
        //Compare new face with saved Faces.
        if (registered.size() > 0) {
            final Pair<String, Float> nearest = findNearest(embeedings[0]);

            if (nearest != null) {
                final String name = nearest.first; // get name of the closest matching face
                distance_local = nearest.second;

                float confidenceVal = 0.0f; // Initialize confidence
                confidenceVal = 1.0f / (1.0f + distance_local);
                String confidenceStr = Float.toString(confidenceVal);

                if (distance_local < distance) {
                    if(isRecognizing)
                        graphicOverlay.updateOverlay(boundingBox, name, confidenceStr);
                    reco_name.setText(name + " \n\nConfidence: " + confidenceStr + "");
                } else {
                    reco_name.setText("Unknown");
                }
            }
        }

    }

    //Compare Faces by distance between face embeddings
    private Pair<String, Float> findNearest(float[] emb) {
        Pair<String, Float> closestMatch = null;

        for (Map.Entry<String, ImageClassification.Recognition> entry : registered.entrySet()) {
            final String name = entry.getKey();
            final float[] knownEmb = ((float[][]) entry.getValue().getExtra())[0];

            float distance = calculateDistance(emb, knownEmb);

            if (closestMatch == null || distance < closestMatch.second) {
                closestMatch = new Pair<>(name, distance);
            }
        }

        return closestMatch;
    }

    private float calculateDistance(float[] emb1, float[] emb2) {
        float distance = 0;

        for (int i = 0; i < emb1.length; i++) {
            float diff = emb1[i] - emb2[i];
            distance += diff * diff;
        }

        return (float) Math.sqrt(distance);
    }
}