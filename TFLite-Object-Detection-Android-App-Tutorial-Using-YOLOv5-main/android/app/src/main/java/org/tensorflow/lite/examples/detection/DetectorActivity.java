/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.DetectorFactory;
import org.tensorflow.lite.examples.detection.tflite.YoloV5Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    // TF_OD_API :  TensorFlow Object Detection API를 사용하는 모드
    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.3f;
    private static final boolean MAINTAIN_ASPECT = true;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 640);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private YoloV5Classifier detector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private BorderedText borderedText;

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        // 디바이스에 맞는 텍스트 사이즈 반환.
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());

        borderedText = new BorderedText(textSizePx); // env/BordedText의 객체를 생성, 텍스트의 색상, 사이즈 결정.
        borderedText.setTypeface(Typeface.MONOSPACE); // 텍스트를 일정한 간격으로 정렬.

        // '객체 추적' 객체를 생성
        tracker = new MultiBoxTracker(this);

        // modelView는 하단 바에서 인공지능 모델 선택 창. 그 중에 선택된 모델의 텍스트 가져오기.
        final int modelIndex = modelView.getCheckedItemPosition();
        final String modelString = modelStrings.get(modelIndex);

        try {
                detector = DetectorFactory.getDetector(getAssets(), modelString); // 모델 지정.
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        int cropSize = detector.getInputSize(); // input_size 반환.

        // 프리뷰 너비와 높이.
        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        // rotation 은 매개변수. getScreenOrientation()은 현재 디바이스의 화면 방향
        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);

        // 이미지의 rgb값을 저장할 비트맵 객체를 생성
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        // 이미지를 프리뷰 프레임에 맞게 자름.
        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        // cropToFrameTransform 은 frameToCropTransform의 역변환 행렬
        // 크롭된 이미지에서 원래의 이미지로 돌아가기 위해 역변환 행렬이 필요함.
        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);


        // 결과 출력 창.
        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        // tracker => MultiBoxTracker.java
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }

    protected void updateActiveModel() {
        // Get UI information before delegating to background
        final int modelIndex = modelView.getCheckedItemPosition(); // 선택된 인공지능 모델
        final int deviceIndex = deviceView.getCheckedItemPosition(); // 선택된 장치(cpu, gpu,,,)
        String threads = threadsTextView.getText().toString().trim(); // threads 텍스트
        final int numThreads = Integer.parseInt(threads);

        handler.post(() -> {
            // 현재 선택된 모델, 디바이스, 쓰레드 수와 이전에 설정된 값들이 같다면, 종료
            if (modelIndex == currentModel && deviceIndex == currentDevice
                    && numThreads == currentNumThreads) {
                return;
            }

            currentModel = modelIndex;
            currentDevice = deviceIndex;
            currentNumThreads = numThreads;

            // Disable classifier while updating
            // 모델 초기화
            if (detector != null) {
                detector.close();
                detector = null;
            }

            // Lookup names of parameters.
            // 문자열로 받아오기
            String modelString = modelStrings.get(modelIndex);
            String device = deviceStrings.get(deviceIndex);

            LOGGER.i("Changing model to " + modelString + " device " + device);

            // Try to load model.

            try {
                detector = DetectorFactory.getDetector(getAssets(), modelString); // 모델 지정.
                // Customize the interpreter to the type of device we want to use.
                if (detector == null) {
                    return;
                }
            }
            catch(IOException e) {
                e.printStackTrace();
                LOGGER.e(e, "Exception in updateActiveModel()");
                Toast toast =
                        Toast.makeText(
                                getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
                toast.show();
                finish();
            }


            // 모델의 디바이스 선택
            if (device.equals("CPU")) {
                detector.useCPU();
            } else if (device.equals("GPU")) {
                detector.useGpu();
            } else if (device.equals("NNAPI")) {
                detector.useNNAPI();
            }

            // 쓰레드 수 설정
            detector.setNumThreads(numThreads);


            int cropSize = detector.getInputSize(); // input_size 리턴.

            // 이미지의 rgb값을 저장할 비트맵 객체를 생성
            croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

            // 이미지를 프리뷰 프레임에 맞게 자름.
            frameToCropTransform =
                    ImageUtils.getTransformationMatrix(
                            previewWidth, previewHeight,
                            cropSize, cropSize,
                            sensorOrientation, MAINTAIN_ASPECT);

            // 역변환 행렬 생성.
            cropToFrameTransform = new Matrix();
            frameToCropTransform.invert(cropToFrameTransform);
        });
    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate(); // 추론 결과를 실시간으로 화면에 반영

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage(); // 흠?
            return;
        }


        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");


        // getRgbBytes()는 rgb이미지 데이터를 가지고 있음.

        //getRgbBytes(): RGB 이미지 데이터를 담고 있는 픽셀 배열
        //0: 픽셀 데이터의 시작 인덱스
        //previewWidth: 픽셀 데이터의 가로 길이
        //0: 비트맵에 설정할 픽셀 데이터의 시작 x 좌표
        //0: 비트맵에 설정할 픽셀 데이터의 시작 y 좌표
        //previewWidth: 비트맵에 설정할 픽셀 데이터의 가로 길이
        //previewHeight: 비트맵에 설정할 픽셀 데이터의 세로 길이

        // getRgbBytes() 에 있는 이미지 rgb데이터를 rgbFrameBitmap에 저장.
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        readyForNextImage();

        // 자른 이미지를 그리기.
        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        // 추론 작업
        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        LOGGER.i("Running detection on image " + currTimestamp);

                        final long startTime = SystemClock.uptimeMillis();

                        // crop된 이미지를 입력으로 추론을 하고 그 결과를 result에 저장. 클래스 레이블 여기서 저장됨.
                        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);

                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                        Log.e("CHECK", "run: " + results.size());

                        // crop 이미지 복사
                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);


                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        // 임계값 설정. MINIMUM_CONFIDENCE_TF_OD_API = 0.3f
                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;

                        // MODE = TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                        }

                        final List<Classifier.Recognition> mappedRecognitions = new LinkedList<Classifier.Recognition>();

                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation(); // 객체의 위치정보를 가져옴.

                            // location이 null이 아니고, result의 신뢰도(정확도)가 최소 임계값(minimumConfidence) 이상인 경우에만 처리
                            if (location != null && result.getConfidence() >= minimumConfidence) {

                                canvas.drawRect(location, paint); // 경계박스를 그림.

                                // result의 위치정보를 mappedRecognitions 에 저장.
                                cropToFrameTransform.mapRect(location);
                                result.setLocation(location);
                                mappedRecognitions.add(result);
                            }
                        }

                        //mappedRecognitions는 최종 추론 결과 중에서 정확도가 일정 이상인 객체들의 리스트입니다.
                        // 이 리스트에는 객체의 위치 정보와 클래스 레이블 등의 정보가 포함되어 있다.

                        // 이미지 내에서 인식된 개체가 어디에 있는지.
                        tracker.trackResults(mappedRecognitions, currTimestamp);
                        trackingOverlay.postInvalidate();

                        computingDetection = false;

                        runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        showFrameInfo(previewWidth + "x" + previewHeight);
                                        showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                                        showInference(lastProcessingTimeMs + "ms");
                                    }
                                });
                    }
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    } // layout xml 파일 지정.

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(() -> detector.setUseNNAPI(isChecked));
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> detector.setNumThreads(numThreads));
    }
}
