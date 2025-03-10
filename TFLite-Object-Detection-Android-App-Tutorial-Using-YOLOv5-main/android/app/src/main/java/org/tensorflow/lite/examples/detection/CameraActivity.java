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

import android.Manifest;
import android.app.Fragment;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Trace;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.util.Size;
import android.view.Surface;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;
import com.google.android.material.bottomsheet.BottomSheetBehavior;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;

import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;

public abstract class CameraActivity extends AppCompatActivity
        implements OnImageAvailableListener,
        Camera.PreviewCallback,
//        CompoundButton.OnCheckedChangeListener,
        View.OnClickListener {
  private static final Logger LOGGER = new Logger();

  private static final int PERMISSIONS_REQUEST = 1;

  private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
  private static final String ASSET_PATH = "";
  protected int previewWidth = 0;
  protected int previewHeight = 0;
  private boolean debug = false;
  protected Handler handler;
  private HandlerThread handlerThread;
  private boolean useCamera2API;
  private boolean isProcessingFrame = false;
  private byte[][] yuvBytes = new byte[3][];
  private int[] rgbBytes = null;
  private int yRowStride;
  protected int defaultModelIndex = 0;
  protected int defaultDeviceIndex = 0;
  private Runnable postInferenceCallback;
  private Runnable imageConverter;
  protected ArrayList<String> modelStrings = new ArrayList<String>();

  private LinearLayout bottomSheetLayout;
  private LinearLayout gestureLayout;
  private BottomSheetBehavior<LinearLayout> sheetBehavior;

  protected TextView frameValueTextView, cropValueTextView, inferenceTimeTextView;
  protected ImageView bottomSheetArrowImageView;
  private ImageView plusImageView, minusImageView;
  protected ListView deviceView;
  protected TextView threadsTextView;
  protected ListView modelView;
  /** Current indices of device and model. */
  int currentDevice = -1;
  int currentModel = -1;
  int currentNumThreads = -1;

  ArrayList<String> deviceStrings = new ArrayList<String>();

  @Override
  protected void onCreate(final Bundle savedInstanceState) {
    LOGGER.d("onCreate " + this);
    super.onCreate(null);

    // 앱에서 화면이 꺼지지 않도록 유지
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

    // layout xml 파일을 tfe_od_activity_camera 으로 지정
    setContentView(R.layout.tfe_od_activity_camera);

    // tool bar 를 설정하고 타이틀은 표시하지 않음.
    Toolbar toolbar = findViewById(R.id.toolbar);
    setSupportActionBar(toolbar);
    //getSupportActionBar().setDisplayShowTitleEnabled(false);

    // 권한이 있는지 확인하고 해당 권한이 없으면 요청
    if (hasPermission()) {
      setFragment();
    } else {
      requestPermission();
    }

    // 언더 바
    threadsTextView = findViewById(R.id.threads); // thread의 숫자 칸
    currentNumThreads = Integer.parseInt(threadsTextView.getText().toString().trim());
    plusImageView = findViewById(R.id.plus); // threads 의 + 버튼
    minusImageView = findViewById(R.id.minus); // threads 의 - 버튼
    deviceView = findViewById(R.id.device_list); // cpu, gpu, nnapi

    deviceStrings.add("CPU");
    deviceStrings.add("GPU");
    deviceStrings.add("NNAPI");

    // ListView에 연결되어 데이터를 표시하고, 사용자가 항목을 선택할 수 있게 함.
    // deviceView 를 단일 선택 할 수 있게 하고 deviceView에 deviceStrings를 연결하여 텍스트 표시.
    deviceView.setChoiceMode(ListView.CHOICE_MODE_SINGLE);
    ArrayAdapter<String> deviceAdapter =
            new ArrayAdapter<>(
                    CameraActivity.this , R.layout.deviceview_row,
                    R.id.deviceview_row_text, deviceStrings);


    // deviceView에 deviceAdapter를 연결하여 데이터를 표시.
    deviceView.setAdapter(deviceAdapter);

    // deviceView에서 기본 장치 인덱스에 해당하는 항목을 선택.
    // defaultDeviceIndex = 0
    deviceView.setItemChecked(defaultDeviceIndex, true);
    currentDevice = defaultDeviceIndex;

    // deviceView의 각 항목이 클릭되었을 때 호출되는 이벤트 리스너를 설정
    // deviceView의 각 항목이 클릭되었을 때 updateActiveModel() 메서드를 호출
    deviceView.setOnItemClickListener(
            new AdapterView.OnItemClickListener() {
              @Override
              public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                updateActiveModel();
              }
            });

    bottomSheetLayout = findViewById(R.id.bottom_sheet_layout);
    gestureLayout = findViewById(R.id.gesture_layout); // 언더 바 당기는  부분
    sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout); // 하단 시트의 동작을 관리
    bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow); // 바 당기는 화살표
    modelView = findViewById((R.id.model_list)); // 인공지능 모델 선택(best-fp16.tflite)

    modelStrings = getModelStrings(getAssets(), ASSET_PATH); // ASSET 디렉터리에 있는 파일들의 이름들을 배열로 저장.

    modelView.setChoiceMode(ListView.CHOICE_MODE_SINGLE);
    ArrayAdapter<String> modelAdapter =
            new ArrayAdapter<>(
                    CameraActivity.this , R.layout.listview_row, R.id.listview_row_text, modelStrings);

    modelView.setAdapter(modelAdapter);
    modelView.setItemChecked(defaultModelIndex, true);
    currentModel = defaultModelIndex;
    modelView.setOnItemClickListener(
            new AdapterView.OnItemClickListener() {
              @Override
              public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                updateActiveModel();
              }
            });

    // gestureLayout 의 이벤트 감시(탐지)
    ViewTreeObserver vto = gestureLayout.getViewTreeObserver();

    vto.addOnGlobalLayoutListener(
            new ViewTreeObserver.OnGlobalLayoutListener() {
              @Override
              public void onGlobalLayout() {
                // 현재 안드로이드 버전이 Jelly Bean보다 낮은지 확인
                // 버전에 따라 removeGlobalOnLayoutListener() 또는
                // removeGOnlobalLayoutListener() 메서드를 호출하여 OnGlobalLayoutListener를 제거
                if (Build.VERSION.SDK_INT < Build.VERSION_CODES.JELLY_BEAN) {
                  gestureLayout.getViewTreeObserver().removeGlobalOnLayoutListener(this);
                }
                else {
                  gestureLayout.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                }
                //                int width = bottomSheetLayout.getMeasuredWidth();

                // gestureLayout의 높이 저장장
               int height = gestureLayout.getMeasuredHeight();

                // 최소 높이(peekHeight)를 height로 설정
                sheetBehavior.setPeekHeight(height);
              }
            });

    // 숨겨질 수 없음
    sheetBehavior.setHideable(false);

    // 하단 시트가 확장됐을 때, 닫혀있을 때에 따라 화살표 그림 바꿈
    sheetBehavior.setBottomSheetCallback(
            new BottomSheetBehavior.BottomSheetCallback() {
              @Override
              public void onStateChanged(@NonNull View bottomSheet, int newState) {
                switch (newState) {
                  case BottomSheetBehavior.STATE_HIDDEN:
                    break;
                  case BottomSheetBehavior.STATE_EXPANDED:
                  {
                    bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_down);
                  }
                  break;
                  case BottomSheetBehavior.STATE_COLLAPSED:
                  {
                    bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                  }
                  break;
                  case BottomSheetBehavior.STATE_DRAGGING:
                    break;
                  case BottomSheetBehavior.STATE_SETTLING:
                    bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                    break;
                }
              }
              @Override
              public void onSlide(@NonNull View bottomSheet, float slideOffset) {}
            });

    frameValueTextView = findViewById(R.id.frame_info); // 하단 시트의 frame
    cropValueTextView = findViewById(R.id.crop_info); // 하단 시트의  crop
    inferenceTimeTextView = findViewById(R.id.inference_info); // 하단 시트의 infernce info

    plusImageView.setOnClickListener(this); //threads의 +버튼 클릭 시 이벤트
    minusImageView.setOnClickListener(this); //threads의 -버튼 클릭 시 이벤트
  }



  // asset 디렉터리 에서 .tflite 인 파일의 이름을 추출
  protected ArrayList<String> getModelStrings(AssetManager mgr, String path){
    ArrayList<String> res = new ArrayList<String>();
    try {
      String[] files = mgr.list(path);
      for (String file : files) {
        String[] splits = file.split("\\.");
        if (splits[splits.length - 1].equals("tflite")) {
          res.add(file);
        }
      }
    }
    catch (IOException e){
      System.err.println("getModelStrings: " + e.getMessage());
    }
    return res;
  }

  // 이미지를 rgb 배열로 반환
  protected int[] getRgbBytes() {
    imageConverter.run();
    return rgbBytes;
  }

  // YUV 형식의 이미지를 RGB로 변환.
  // YUV 형식 : 밝기 정보(Y)와 색상 정보(UV)로 구성. 캡처 이미지가 YUV형식 일 수도 있음.
  protected int getLuminanceStride() {
    return yRowStride;
  }
  protected byte[] getLuminance() {
    return yuvBytes[0];
  }

  /** Callback for android.hardware.Camera API */

  // 카메라에서 전달된 프레임 데이터를 받아와서 RGB 형식으로 변환하고
  // 후속 처리를 위해 필요한 변수들을 설정하는 역할
  // 실시간 카메라 프리뷰를 데이터로 사용
  @Override
  public void onPreviewFrame(final byte[] bytes, final Camera camera) {
    // 현재 프레임이 처리 중인지 확인
    // 이미 처리 중인 경우에는 프레임을 건너뛰고 함수를 종료
    if (isProcessingFrame) {
      LOGGER.w("Dropping frame!");
      return;
    }

    try {
      // Initialize the storage bitmaps once when the resolution is known.
      // rgbBytes 배열이 초기화되어 있는지 확인
      if (rgbBytes == null) {
        // 프레임의 사이즈를 얻어옴
        Camera.Size previewSize = camera.getParameters().getPreviewSize();
        previewHeight = previewSize.height;
        previewWidth = previewSize.width;

        // 프레임 사이즈에 맞게 배열 초기화
        rgbBytes = new int[previewWidth * previewHeight];

        onPreviewSizeChosen(new Size(previewSize.width, previewSize.height), 90);
      }
    } catch (final Exception e) {
      LOGGER.e(e, "Exception!");
      return;
    }

    isProcessingFrame = true;
    yuvBytes[0] = bytes; //  YUV 데이터를 받아와 yuvBytes[0] 배열에 저장
    yRowStride = previewWidth;

    imageConverter =
            new Runnable() {
              @Override
              public void run() {
                // YUV 형식의 이미즈를 RGB 형식으로 변환 후, rgbBytes배열에 저장.
                ImageUtils.convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes);
              }
            };


    postInferenceCallback =
            new Runnable() {
              @Override
              public void run() {
                // 전 프리뷰 프레임의 바이트 버퍼를 반환하고, 다음 프리뷰 프레임을 받을 준비를 하는 역할.
                // 현재 처리 중인 프리뷰 프레임의 바이트 버퍼를 반환하는 역할을 합니다.
                // 이렇게 반환된 바이트 버퍼는 다음 프레임을 받을 때 재사용됩니다.
                // 프리뷰(Preview) : 카메라로부터 실시간으로 받아온 이미지를 화면에 보여주는 것을 말함.
                camera.addCallbackBuffer(bytes);
                isProcessingFrame = false;
              }
            };


    processImage();
  }

  /** Callback for Camera2 API */
  //  ImageReader로부터 받은 YUV 이미지를 RGB 형식으로 변환하여 처리할 수 있게 됩니다.
  // 저장된 이미지를 데이터로 사용
  @Override
  public void onImageAvailable(final ImageReader reader) {
    // We need wait until we have some size from onPreviewSizeChosen

    // 프리뷰이미지의 크기가 설정되지 않았거나 0인 경우, 함수 종료
    if (previewWidth == 0 || previewHeight == 0) {
      return;
    }

    // rgbBytes 배열이 초기화되지 않은 경우, 프리뷰 크기에 맞게 초기화합.
    // rgbBytes는 RGB 형식의 이미지를 저장할 예정.
    if (rgbBytes == null) {
      rgbBytes = new int[previewWidth * previewHeight];
    }

    try {
      // ImageReader로부터 (실시간) 최신 이미지(image)를 획득.
      // final : 값을 변경할 수 없음.
      final Image image = reader.acquireLatestImage();


      // 이미지가 없는 경우 메서드를 종료.
      if (image == null) {
        return;
      }

      // 이미지 처리 중인 경우, 종료
      // 이미지 처리 작업은 CPU나 GPU 등의 자원을 사용하며, 시간이 오래 걸릴 수 있음.
      // 따라서 이미지 처리 중인 경우에는 다른 이미지가 도착하더라도
      // 현재 이미지 처리 작업이 완료될 때까지 기다리는 것이 효율적임.
      // 이미지 처리 작업이 중복으로 실행되지 않도록 하기 위해 이미지가 처리 중인 경우,
      // 현재 도착한 이미지를 닫고 함수 실행을 종료하여 중복 처리를 방지함.
      // 이를 통해 이미지 처리 작업의 순서를 보장하고, 정확한 결과를 얻을 수 있음.
      if (isProcessingFrame) {
        image.close();
        return;
      }

      isProcessingFrame = true;

      Trace.beginSection("imageAvailable");

      final Plane[] planes = image.getPlanes(); //  YUV 이미지의 데이터를 추출
      fillBytes(planes, yuvBytes); // YUV 이미지 데이터를 yuvBytes 배열에 저장.
      yRowStride = planes[0].getRowStride(); // Y 데이터의 행 간격, 행 간격은 한 행의 데이터 크기와 다음 행의 데이터 시작 위치 사이의 거리를 의미
      final int uvRowStride = planes[1].getRowStride(); // UV 데이터의 행 간격
      final int uvPixelStride = planes[1].getPixelStride(); // UV 데이터의 픽셀 간격

      imageConverter =
              new Runnable() {
                @Override
                public void run() {
                  // YUV 이미지를 RGB 형식으로 변환하는 작업.
                  ImageUtils.convertYUV420ToARGB8888(
                          yuvBytes[0], // Y 플레인 데이터
                          yuvBytes[1], //  U 플레인 데이터
                          yuvBytes[2], // V 플레인 데이터
                          previewWidth, // 프리뷰 이미지의 가로 해상도
                          previewHeight, // 프리뷰 이미지의 세로 해상도
                          yRowStride, // Y 플레인의 행 간격
                          uvRowStride,  // UV 플레인의 행 간격
                          uvPixelStride, //  UV 플레인의 픽셀 간격
                          rgbBytes); //  RGB8888 형식의 결과 이미지를 저장할 배열
                }
              };

      // 추론 작업이 완료되었을 때 실행되는 콜백 함수
      postInferenceCallback =
              new Runnable() {
                @Override
                public void run() {
                  image.close();
                  isProcessingFrame = false; // false로 설정하여 이미지 처리가 완료되었음을 나타냄.
                }
              };

      processImage();

    } catch (final Exception e) {
      LOGGER.e(e, "Exception!");
      Trace.endSection();
      return;
    }
    Trace.endSection();
  }

  @Override
  public synchronized void onStart() {
    LOGGER.d("onStart " + this);
    super.onStart();
  }

  @Override
  public synchronized void onResume() {
    LOGGER.d("onResume " + this);
    super.onResume();

    handlerThread = new HandlerThread("inference");
    handlerThread.start();
    handler = new Handler(handlerThread.getLooper());
  }

  @Override
  public synchronized void onPause() {
    LOGGER.d("onPause " + this);

    handlerThread.quitSafely();
    try {
      handlerThread.join();
      handlerThread = null;
      handler = null;
    } catch (final InterruptedException e) {
      LOGGER.e(e, "Exception!");
    }

    super.onPause();
  }

  @Override
  public synchronized void onStop() {
    LOGGER.d("onStop " + this);
    super.onStop();
  }

  @Override
  public synchronized void onDestroy() {
    LOGGER.d("onDestroy " + this);
    super.onDestroy();
  }

  protected synchronized void runInBackground(final Runnable r) {
    if (handler != null) {
      handler.post(r);
    }
  }

  //  메소드는 권한 요청 결과를 확인
  @Override
  public void onRequestPermissionsResult(
          final int requestCode, final String[] permissions, final int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == PERMISSIONS_REQUEST) {
      if (allPermissionsGranted(grantResults)) {
        setFragment();
      } else {
        requestPermission();
      }
    }
  }

  // grantResults 배열에 포함된 모든 권한이 허용되었는지 확인하는 데 사용되는 메소드입니다.
  private static boolean allPermissionsGranted(final int[] grantResults) {
    for (int result : grantResults) {
      if (result != PackageManager.PERMISSION_GRANTED) {
        return false;
      }
    }
    return true;
  }

  // 현재 앱이 카메라 권한을 가지고 있는지를 확인함.
  private boolean hasPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
    } else {
      return true;
    }
  }

  // 카메라 권한을 요청
  private void requestPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
        Toast.makeText(
                        CameraActivity.this,
                        "Camera permission is required for this demo",
                        Toast.LENGTH_LONG)
                .show();
      }
      requestPermissions(new String[] {PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
    }
  }

  // Returns true if the device supports the required hardware level, or better.
  private boolean isHardwareLevelSupported(
          CameraCharacteristics characteristics, int requiredLevel) {
    int deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
    if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
      return requiredLevel == deviceLevel;
    }
    // deviceLevel is not LEGACY, can use numerical sort
    return requiredLevel <= deviceLevel;
  }


  // 사용 가능한 카메라 선택
  private String chooseCamera() {
    final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
    try {
      for (final String cameraId : manager.getCameraIdList()) {
        final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

        // We don't use a front facing camera in this sample.
        final Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
        if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
          continue;
        }

        final StreamConfigurationMap map =
                characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

        if (map == null) {
          continue;
        }

        // Fallback to camera1 API for internal cameras that don't have full support.
        // This should help with legacy situations where using the camera2 API causes
        // distorted or otherwise broken previews.
        useCamera2API =
                (facing == CameraCharacteristics.LENS_FACING_EXTERNAL)
                        || isHardwareLevelSupported(
                        characteristics, CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL);
        LOGGER.i("Camera API lv2?: %s", useCamera2API);
        return cameraId;
      }
    } catch (CameraAccessException e) {
      LOGGER.e(e, "Not allowed to access camera");
    }

    return null;
  }

// 카메라 프래그먼트는 카메라 API와 상호작용하고 이미지 처리 및 분석을 위한 데이터를 제공하는 역할을 수행
  protected void setFragment() {
    // 사용할 카메라 id를 가져옴
        String cameraId = chooseCamera();

        Fragment fragment;
        // useCamera2API : 카메라를 선택할 때 Camera2 API를 지원하는 기기인지확인하는 boolean 변수
        if (useCamera2API) {
          CameraConnectionFragment camera2Fragment =
                  // CameraConnectionFragment 를 생성
              CameraConnectionFragment.newInstance(
                      new CameraConnectionFragment.ConnectionCallback() {
                        @Override
                        public void onPreviewSizeChosen(final Size size, final int rotation) {
                          previewHeight = size.getHeight();
                          previewWidth = size.getWidth();
                          CameraActivity.this.onPreviewSizeChosen(size, rotation);
                        }
                      },
                      this,
                      getLayoutId(),
                      getDesiredPreviewFrameSize());

      camera2Fragment.setCamera(cameraId); // 카메라 id 설정
      fragment = camera2Fragment;
    }
        else {
          // LegacyCameraConnectionFragment를 생성
      fragment = new LegacyCameraConnectionFragment(this, getLayoutId(), getDesiredPreviewFrameSize());
    }

    // 생성한 프래그먼트를 replace() 메소드를 사용하여 R.id.container 뷰에 배치하고, 트랜잭션을 커밋하여 화면에 표시합니다.
    getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
  }

  // Plane 배열에서 YUV 이미지 데이터를 추출하여 yuvBytes 배열에 채우는 역할
  protected void fillBytes(final Plane[] planes, final byte[][] yuvBytes) {
    // Because of the variable row stride it's not possible to know in
    // advance the actual necessary dimensions of the yuv planes.
    for (int i = 0; i < planes.length; ++i) {
      final ByteBuffer buffer = planes[i].getBuffer();
      if (yuvBytes[i] == null) {
        LOGGER.d("Initializing buffer %d at size %d", i, buffer.capacity());
        yuvBytes[i] = new byte[buffer.capacity()];
      }
      buffer.get(yuvBytes[i]);
    }
  }

  public boolean isDebug() {
    return debug;
  }

  protected void readyForNextImage() {
    if (postInferenceCallback != null) {
      postInferenceCallback.run();
    }
  }

  // 현재 디바이스의 화면 방향. 화면 방향에 따라 이미지 회전.
  protected int getScreenOrientation() {
    switch (getWindowManager().getDefaultDisplay().getRotation()) {
      case Surface.ROTATION_270:
        return 270;
      case Surface.ROTATION_180:
        return 180;
      case Surface.ROTATION_90:
        return 90;
      default:
        return 0;
    }
  }

//  @Override
//  public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
//    setUseNNAPI(isChecked);
//    if (isChecked) apiSwitchCompat.setText("NNAPI");
//    else apiSwitchCompat.setText("TFLITE");
//  }


  // 쓰레드를 증가시키면 모델의 추론 속도가 향상될 수 있습니다.
  // 여러 개의 스레드를 사용하여 모델의 작업을 병렬로 처리할 수 있기 때문에 처리 속도가 향상될 수 있습니다.
  @Override
  public void onClick(View v) {
    // 클릭된 버튼이 plus 버튼인지 확인.
    if (v.getId() == R.id.plus) {
      // 현재 스레드 수를 증가시키고 증가된 스레드 수를 표시.
      String threads = threadsTextView.getText().toString().trim();
      int numThreads = Integer.parseInt(threads);
      if (numThreads >= 9) return;
      numThreads++;
      threadsTextView.setText(String.valueOf(numThreads));
      setNumThreads(numThreads);
    }
    // 클릭된 버튼이 minus 버튼인지 확인.
    else if (v.getId() == R.id.minus) {
      // 현재 스레드 수를 감소시키고 감소된 스레드 수를 표시.
      String threads = threadsTextView.getText().toString().trim();
      int numThreads = Integer.parseInt(threads);
      if (numThreads == 1) {
        return;
      }
      numThreads--;
      threadsTextView.setText(String.valueOf(numThreads));
      setNumThreads(numThreads);
    }
  }

  protected void showFrameInfo(String frameInfo) {
    frameValueTextView.setText(frameInfo);
  }

  protected void showCropInfo(String cropInfo) {
    cropValueTextView.setText(cropInfo);
  }

  protected void showInference(String inferenceTime) {
    inferenceTimeTextView.setText(inferenceTime);
  }

  protected abstract void updateActiveModel();
  protected abstract void processImage();

  protected abstract void onPreviewSizeChosen(final Size size, final int rotation);

  protected abstract int getLayoutId();

  protected abstract Size getDesiredPreviewFrameSize();

  protected abstract void setNumThreads(int numThreads);

  protected abstract void setUseNNAPI(boolean isChecked);
}
