/**
 * 
 */
package edu.cmu.cs211.seamcarving;

import java.util.Vector;

/**
 * Class that is used to find the minimum energy vertical seam in an image
 * 
 */
public class SeamFinder {

  //mathematically needed in the recurrence
  private static final float INFINITY = Float.MAX_VALUE;
	
  public SeamFinder() {

  }

  /**
   * Finds the minimum vertical seam in the cost image. A seam is a connected
   * line through the image.
   * 
   * @param costs The cost of removing each pixel in the image
   * @return A vector the same height as the image where each entry specifies,
   *         for a single row, which column to remove. So, if vector[5] = 3,
   *         then the minimum seam goes through the pixel at row 5 and column 3.
   */
  public Vector<Integer> findMinSeam(GrayImage costs) {
	  //generate a minimum cost table so that we can find the minimum
	  float[][] minCosts = getMinCostsTable(costs);
	  
	  //find minimum cell in bottom row and work backwards
	  //to generate the vector
	  Vector<Integer> seam = getSeam(minCosts);
	  
	  return seam;
  }

  /**
   * Works backwards from bottom to find the seam with the minimum
   * cost. Iterates through the cost table upwards looking for the minimum
   * of the 3 adjacent cells and adds them in reverse order to the vector
   * @param minCosts the minimum cost table
   * @return
   */
  private Vector<Integer> getSeam(float[][] minCosts) {
	  Vector<Integer> seam = new Vector<Integer>();
	  int m = minCosts.length;
	  int n = minCosts[0].length;
	  
	  //initialize minimums
	  float min = minCosts[m-1][0];
	  int minIndex = 0;
	  
	  //find minimum in bottom row
	  for(int j = 1; j < n; j++) {
		  if(minCosts[m-1][j] < min) {
			  min = minCosts[m-1][j];
			  minIndex = j;
		  }
	  }
	  seam.insertElementAt(minIndex, 0);
	  
	  //work upwards from bottom (use minimum of 3 adjacent cells above)
	  for(int i = m-2; i >= 0; i--) {
		  float left = INFINITY, right = INFINITY;
		  float middle = minCosts[i][minIndex];
		  
		  //away from left edge: include left-most term
		  if(minIndex > 0)
			  left = minCosts[i][minIndex-1];
		  //away from right edge: include right-most term
		  if(minIndex < n-1)
			  right = minCosts[i][minIndex+1];
		  
		  if(left < middle && left < right) //if left is min, move left
			  minIndex--;
		  else if(right < middle && right < left) //if right is min, move right
			  minIndex++;
		  //else do not change column
		  
		  //insert coordinate into vector in reverse order
		  seam.insertElementAt(minIndex, 0);
	  }
	  
	  return seam;
  }
  
  /**
   * Generates the minimum cost table for a seam ending at cell (i, j)
   * @param costs contains the gradient values
   * @return a summed minCost table
   */
  private float[][] getMinCostsTable(GrayImage costs) {
	  //grab dimensions and make new table
	  int m = costs.getHeight();
	  int n = costs.getWidth();
	  float[][] minCosts = new float[m][n];
	  
	  //fill in table using bottom-up dynamic programming
	  for(int i = 0; i < m; i++)
		  for(int j = 0; j < n; j++)
			  minCosts[i][j] = getMinCost(costs, minCosts, i, j);
	  
	  return minCosts;
  }
  
  /**
   * Returns the minimum cost to get to cell (i, j)
   * Uses the recurrence:
   * base cases:
   * minCost = cost(i, j) for i = 0
   * minCost = infinity   for j < 0
   * minCost = infinity   for j >= n
   * minCost(i, j) = cost(i, j) +
   *                    minimum{
   *                      minCost(i-1, j-1)
   *                      minCost(i-1, j+0)
   *                      minCost(i-1, j+1)
   *                    }
   */
  private float getMinCost(GrayImage costs, float[][] minCosts,
		                   int i, int j) {	  
	  //top row: we just return the cost at (i, j)
	  if(i == 0)
		  return costs.get(i, j);
	  
	  //initialize outer terms to infinity
	  int n = costs.getWidth();
	  float left = INFINITY, right = INFINITY;
	  float middle = minCosts[i-1][j];
	  
	  //away from left edge: include left-most term
	  if(j > 0)
		  left = minCosts[i-1][j-1];
	  //away from right edge: include right-most term
	  if(j < n-1)
		  right = minCosts[i-1][j+1];
	  
	  //take the minimum of 3 terms in above row, add cost of current cell
	  float min = Math.min(left, Math.min(middle, right));
	  return min + costs.get(i, j);
  }
  
}
