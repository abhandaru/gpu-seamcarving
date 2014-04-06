/**
 * 
 */
package edu.cmu.cs211.seamcarving;

/**
 * A filter object that calculate the magnitude of the gradient of a GrayImage.
 * 
 * The magnitude of the gradient at a point is defined as:
 * 
 * g(i,j) = sqrt( ( p(i,j+1)-p(i,j) ) ^2 + ( p(i+1,j)-p(i,j) ) ^2 )
 * 
 * where p(i,j) is the value of the pixel at (i,j)
 * 
 * Any point outside of the frame is considered to have a pixel value 0
 *
 */
public class GradientFilter {
  
  public GradientFilter() {
    
  }
  
  /**
   * Apply filter to the image
   * 
   * @param image Image to filter
   * @return A resulting image of the gradient magnitude
   */
  public GrayImage filter(GrayImage image) {
	  //grab dimensions, initialize new data array
	  int m = image.getHeight();
	  int n = image.getWidth();
	  float[][] grad = new float[m][n];
	  
	  //fill the gradient array
	  for(int i = 0; i < m; i++)
		  for(int j = 0; j < n; j++)
			  grad[i][j] = gradOfPixel(image, i, j);
	  
	  return new GrayImage(grad);
  }
  
  /**
   * Grab the gradient value at any index of the image
   * @param img GrayImage passed through filter function
   * @param i row index of gradPixel to be determined
   * @param j column index
   * @return the gradient value at those indices
   */
  private float gradOfPixel(GrayImage img, int i, int j) {
	  //apply the formula
	  float grad = (float) Math.sqrt(
			  Math.pow(get(img, i,j+1) - get(img, i,j), 2) + 
			  Math.pow(get(img, i+1,j) - get(img, i,j), 2)
			);
	  
	  return grad;
  }
  
  /**
   * Gradient helper method. Grabs pixel value for valid indices
   * Returns 0 if the indices are out of bounds
   * @return float pixel value
   */
  private float get(GrayImage img, int i, int j) {
	  int m = img.getHeight();
	  int n = img.getWidth();
	  
	  //deal with pixels out of the image
	  if(i < 0 || i >= m) return 0;
	  if(j < 0 || j >= n) return 0;
	  
	  return img.get(i, j);
  }

}
