/**
 * 
 */
package edu.cmu.cs211.seamcarving;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.util.Vector;

/**
 * This filter implements the seam carving algorithm to resize an image. The
 * algorithm picks a continuous seam of uninteresting pixels and
 * removes them in order to resize images.
 * 
 * This is a simplified implementation that only does one column at a time.
 * 
 * @article @inproceedings{AS-SeamCarving-07, title = {Seam Carving for
 *          Content-Aware Image Resizing}, author = {Shai Avidan and Ariel
 *          Shamir}, journal = {ACM Transactions on Graphics, (Proceedings
 *          SIGGRAPH 2007)}, volume = {26}, number = {3}, year = {2007} }
 * 
 * To see a cool demo, go to: {@link http://www.youtube.com/watch?v=vIFCV2spKtg}
 * 
 */
public class SeamCarvingFilter {

  public SeamCarvingFilter() {  }

  /**
   * Uses seam carving to remove a single column from the image
   * 
   * @param original Image to remove column from
   * @return A new image that is one column smaller than the original
   * @throws IllegalArgumentException if the original image is invalid or too
   *           small
   * @throws NullPointerException if the original image pointer is null
   */
  public BufferedImage removeColumn(final BufferedImage original)
      throws IllegalArgumentException, NullPointerException {
    int newWidth = original.getWidth() - 1;
    if (newWidth < 0) {
      throw new IllegalArgumentException(
          "The image must be at least one column wide");
    }

    // Create the return image
    BufferedImage retImage = new BufferedImage(original.getWidth() - 1,
        original.getHeight(), original.getType());
    WritableRaster raster = retImage.getRaster();

    Vector<Integer> seam = findSeam(original);

    // Now fill in the new image one row at a time
    final Raster origData = original.getData();
    float[] garbArray = new float[original.getWidth() * 3]; // Garbage array that's
                                                        // needed by the
                                                        // getPixels function
    for (int row = 0; row < seam.size(); ++row) {
      int curCol = seam.get(row);
      if (curCol > 0) {
        // There are pixels to copy to the left of the seam
        raster.setPixels(0, row, curCol, 1, origData.getPixels(0, row,
            curCol, 1, garbArray));
      }

      if (seam.get(row) < original.getWidth() - 1) {
        // There are pixels to copy to the right of the seam
        int widthAfter = retImage.getWidth() - curCol;
        raster.setPixels(curCol, row, widthAfter, 1, origData
            .getPixels(curCol + 1, row, widthAfter, 1,
                garbArray));
      }
    }

    return retImage;
  }

  /**
   * Determines the seam with the minimum cost
   * 
   * @param original Image to calculate the seam for
   * @return column coordinates for each pixel in the seam one for each row in
   *         order
   */
  private Vector<Integer> findSeam(final BufferedImage original) {
    
    // Calculate the magnitude gradient of the image
    GrayImage grayImage = new GrayImage(original);
    grayImage = new GradientFilter().filter(grayImage);
    

    // Use DP to find the seam
    SeamFinder seamFinder = new SeamFinder();
    
    
    return seamFinder.findMinSeam(grayImage);
  }

}
