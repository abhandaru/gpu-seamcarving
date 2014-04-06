package edu.cmu.cs211.seamcarving;

import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.awt.image.Raster;

/**
 * Object that encapsulates a grayscale image. We're not using the built in Java
 * image libraries because they handle a lot more complicated stuff and are thus
 * harder to understand
 * 
 * @author Mark Desnoyer
 */
public class GrayImage {
  
  private float[][] data_;  // The data in [row][column] format
  
  /**
   * Creates this object from a BufferedImage in the GUI
   * 
   * @param image 
   */
  public GrayImage(final BufferedImage image) {
    BufferedImage grayImage = image;
    
    if (image.getType() != BufferedImage.TYPE_BYTE_GRAY || image.getType() != BufferedImage.TYPE_USHORT_GRAY) {
      // Turn the image into a grascale image
      grayImage = new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_GRAY), null).filter(image, null);
    }


    if (grayImage == image)
        return;
    
    // Now copy the data into our array
    float[] curRow = new float[image.getWidth()];
    data_ = new float[image.getHeight()][image.getWidth()];
    Raster raster = grayImage.getData();
    for (int i = 0; i < image.getHeight(); ++i) {
      curRow = raster.getSamples(0, i, image.getWidth(), 1, 0, curRow);
      data_[i] = curRow.clone();
    }
  }
  
  /**
   * Constructor where we already have the data array
   * 
   * @param image
   */
  public GrayImage(float[][] image) {
    if (image == null)
      throw new NullPointerException("Invalid image data");
    data_ = image.clone();
  }
  
  /**
   * Returns the value of a pixel
   * 
   * @param row Row to get the pixel from
   * @param col Column to get the pixel from
   * @return The value of the pixel at (row, col)
   */
  public float get(int row, int col) {
    return data_[row][col];
  }
  
  /**
   * Sets the value of a pixel
   * 
   * @param row Row of the pixel to set
   * @param col Column of the pixel to set
   * @param value The value at (row, col)
   */
  public void set(int row, int col, float value) {
    data_[row][col] = value;
  }
  
  /**
   * @return The height of the image
   */
  public int getHeight() { return data_.length; }
  
  /**
   * @return The width of the image
   */
  public int getWidth() {
    if (getHeight() > 0) {
      return data_[0].length;
    }
    return 0;
  }

}
