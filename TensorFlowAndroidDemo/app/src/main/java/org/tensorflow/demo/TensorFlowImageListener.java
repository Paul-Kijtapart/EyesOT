/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.AsyncTask;
import android.os.Handler;
import android.os.Trace;

import junit.framework.Assert;

import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;

import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.net.URL;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.net.ssl.HttpsURLConnection;

/**
 * Class that takes in preview frames and converts the image to Bitmaps to process with Tensorflow.
 */
public class TensorFlowImageListener implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  private static final boolean SAVE_PREVIEW_BITMAP = false;

  // These are the settings for the original v1 Inception model. If you want to
  // use a model that's been produced from the TensorFlow for Poets codelab,
  // you'll need to set IMAGE_SIZE = 299, IMAGE_MEAN = 128, IMAGE_STD = 128,
  // INPUT_NAME = "Mul:0", and OUTPUT_NAME = "final_result:0".
  // You'll also need to update the MODEL_FILE and LABEL_FILE paths to point to
  // the ones you produced.
  private static final int NUM_CLASSES = 1001;
  private static final int INPUT_SIZE = 224;
  private static final int IMAGE_MEAN = 117;
  private static final float IMAGE_STD = 1;
  private static final String INPUT_NAME = "input:0";
  private static final String OUTPUT_NAME = "output:0";

  private static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
  private static final String LABEL_FILE =
      "file:///android_asset/imagenet_comp_graph_label_strings.txt";

  private Integer sensorOrientation;

  private final TensorFlowClassifier tensorflow = new TensorFlowClassifier();

  private int previewWidth = 0;
  private int previewHeight = 0;
  private byte[][] yuvBytes;
  private int[] rgbBytes = null;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;

  private boolean computing = false;
  private Handler handler;

  private RecognitionScoreView scoreView;

  Set<String> objectsFound;
  Set<String> dangerousThings;

  public void initialize(
      final AssetManager assetManager,
      final RecognitionScoreView scoreView,
      final Handler handler,
      final Integer sensorOrientation) {
    Assert.assertNotNull(sensorOrientation);
    tensorflow.initializeTensorFlow(
        assetManager, MODEL_FILE, LABEL_FILE, NUM_CLASSES, INPUT_SIZE, IMAGE_MEAN, IMAGE_STD,
        INPUT_NAME, OUTPUT_NAME);
    this.scoreView = scoreView;
    this.handler = handler;
    this.sensorOrientation = sensorOrientation;
    this.objectsFound = new HashSet<>();
    this.dangerousThings = new HashSet<>();
    this.dangerousThings.add("rifle");
    this.dangerousThings.add("cleaver");
    this.dangerousThings.add("hatchet");
    this.dangerousThings.add("hammer");
    this.dangerousThings.add("chain saw");
    this.dangerousThings.add("assult rifle");
  }

  private void drawResizedBitmap(final Bitmap src, final Bitmap dst) {
    Assert.assertEquals(dst.getWidth(), dst.getHeight());
    final float minDim = Math.min(src.getWidth(), src.getHeight());

    final Matrix matrix = new Matrix();

    // We only want the center square out of the original rectangle.
    final float translateX = -Math.max(0, (src.getWidth() - minDim) / 2);
    final float translateY = -Math.max(0, (src.getHeight() - minDim) / 2);
    matrix.preTranslate(translateX, translateY);

    final float scaleFactor = dst.getHeight() / minDim;
    matrix.postScale(scaleFactor, scaleFactor);

    // Rotate around the center if necessary.
    if (sensorOrientation != 0) {
      matrix.postTranslate(-dst.getWidth() / 2.0f, -dst.getHeight() / 2.0f);
      matrix.postRotate(sensorOrientation);
      matrix.postTranslate(dst.getWidth() / 2.0f, dst.getHeight() / 2.0f);
    }

    final Canvas canvas = new Canvas(dst);
    canvas.drawBitmap(src, matrix, null);
  }

  @Override
  public void onImageAvailable(final ImageReader reader) {
    Image image = null;
    try {
      image = reader.acquireLatestImage();

      if (image == null) {
        return;
      }

      // No mutex needed as this method is not reentrant.
      if (computing) {
        image.close();
        return;
      }
      computing = true;

      Trace.beginSection("imageAvailable");

      final Plane[] planes = image.getPlanes();

      // Initialize the storage bitmaps once when the resolution is known.
      if (previewWidth != image.getWidth() || previewHeight != image.getHeight()) {
        previewWidth = image.getWidth();
        previewHeight = image.getHeight();

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbBytes = new int[previewWidth * previewHeight];
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Config.ARGB_8888);

        yuvBytes = new byte[planes.length][];
        for (int i = 0; i < planes.length; ++i) {
          yuvBytes[i] = new byte[planes[i].getBuffer().capacity()];
        }
      }

      for (int i = 0; i < planes.length; ++i) {
        planes[i].getBuffer().get(yuvBytes[i]);
      }

      final int yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();
      ImageUtils.convertYUV420ToARGB8888(
          yuvBytes[0],
          yuvBytes[1],
          yuvBytes[2],
          rgbBytes,
          previewWidth,
          previewHeight,
          yRowStride,
          uvRowStride,
          uvPixelStride,
          false);

      image.close();
    } catch (final Exception e) {
      if (image != null) {
        image.close();
      }
      LOGGER.e(e, "Exception!");
      Trace.endSection();
      return;
    }

    rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
    drawResizedBitmap(rgbFrameBitmap, croppedBitmap);

    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    handler.post(
        new Runnable() {
          @Override
          public void run() {
            final List<Classifier.Recognition> results = tensorflow.recognizeImage(croppedBitmap);

            LOGGER.v("%d results", results.size());
            for (final Classifier.Recognition result : results) {
              LOGGER.v("Result: " + result.getTitle());
              String title = result.getTitle();
              if (dangerousThings.contains(title) && !objectsFound.contains(title)) {
                objectsFound.add(title);
                try {
                  double lat = 49.277138;
                  double lon = -123.117802;
                  long timestamp = System.currentTimeMillis();
                  String device_id = "PPAP01";
                  String type = "Image";
                  String data = title;
                  double confidence = result.getConfidence();

                  JSONObject postData = new JSONObject();
                  postData.put("lat", lat);
                  postData.put("lon", lon);
                  postData.put("timestamp", timestamp);
                  postData.put("device_id", device_id);
                  postData.put("type", type);
                  postData.put("data", data);
                  postData.put("confidence", confidence);

                  new SendDeviceDetails().execute("http://eyesofthethings.herokuapp.com/newEvent", postData.toString());
                  LOGGER.i("sent " + postData.toString());
                } catch (JSONException e) {
                  e.printStackTrace();
                }
              }
            }
            scoreView.setResults(results);
            computing = false;
          }
        });

    Trace.endSection();
  }

  private class SendDeviceDetails extends AsyncTask<String, Void, String> {

    @Override
    protected String doInBackground(String... params) {

      String data = "";

      HttpURLConnection httpURLConnection = null;
      try {

        httpURLConnection = (HttpURLConnection) new URL(params[0]).openConnection();
        httpURLConnection.setRequestMethod("POST");
        httpURLConnection.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
        httpURLConnection.setDoOutput(true);

        DataOutputStream wr = new DataOutputStream(httpURLConnection.getOutputStream());
        wr.writeBytes(params[1]);
        wr.flush();
        wr.close();

        InputStream in = httpURLConnection.getInputStream();
        InputStreamReader inputStreamReader = new InputStreamReader(in);

        int inputStreamData = inputStreamReader.read();
        while (inputStreamData != -1) {
          char current = (char) inputStreamData;
          inputStreamData = inputStreamReader.read();
          data += current;
        }
      } catch (Exception e) {
        e.printStackTrace();
      } finally {
        if (httpURLConnection != null) {
          httpURLConnection.disconnect();
        }
      }

      return data;
    }

    @Override
    protected void onPostExecute(String result) {
      super.onPostExecute(result);
    }
  }
}
