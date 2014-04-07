/**
 * 
 */
package edu.cmu.cs211.seamcarving.tests;


import static org.junit.Assert.*;

import java.util.Vector;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import edu.cmu.cs211.seamcarving.GrayImage;
import edu.cmu.cs211.seamcarving.SeamFinder;


public class SeamFinderTest {
  
  protected SeamFinder finder_;

  /**
   * @throws java.lang.Exception
   */
  @Before
  public void setUp() throws Exception {
    finder_ = new SeamFinder();
  }
  
  /**
   * Utility to check that given an image of costs, we get the resulting seam solution
   * 
   * @param costs Array of costs representing the image in [row][column] format
   * @param expectedSeam Array of columns that should be selected in the seam, one for each row. 
   */
  protected void testSeam(float[][] costs, int[] expectedSeam) {
    Vector<Integer> seam = finder_.findMinSeam(new GrayImage(costs));
    assertSeamEquals(seam, expectedSeam);
  }
  
  protected void assertSeamEquals(Vector<Integer> seam, int[] expected) {
    assertEquals("The seam is the wrong length. Should be the same as the number of rows. ", expected.length, seam.size());
    
    for (int i = 0; i < expected.length; ++i) {
      assertEquals("Wrong pixel chosen in the seam for row " + i + ": ", expected[i], seam.get(i).intValue());
    }
  }
  
  @Test
  public void verticalSeamTest() {
    // Testing when the best seam is going to be completely vertical in the center of the image
    float[][] array = {
        {1, 0, 1},
        {2, 0, 2},
        {1, 0, 1},
        {3, 0, 3}
        };
    
    testSeam(array, new int[]{1,1,1,1});
  }
  
  @Test
  public void rightSeamTest() {
    // Testing when the best seam is going to be completely vertical in the center of the image
    float[][] array = {
        {1, 1, 0},
        {2, 2, 0},
        {1, 1, 0},
        {3, 3, 0}
        };
    
    testSeam(array, new int[]{2,2,2,2});
  }
  
  @Test
  public void leftSeamTest() {
    // Testing when the best seam is going to be completely vertical in the center of the image
    float[][] array = {
        {0, 1, 1},
        {0, 2, 2},
        {0, 1, 1},
        {0, 3, 3}
        };
    
    testSeam(array, new int[]{0,0,0,0});
  }
  
  @Test
  public void negativeCostsTest() {
    // Testing when the best seam is going to be completely vertical in the center of the image
    float[][] array = {
        {-1, 1, 2},
        {2, -2, 2},
        {-1, 1, 1},
        {3, -3, 5}
        };
    
    testSeam(array, new int[]{0,1,0,1});
  }
}
