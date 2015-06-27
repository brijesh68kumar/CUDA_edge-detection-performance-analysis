import java.awt.*;
import java.lang.Math.*;
import java.awt.Color.*;

public class Convolution{
	/*public static void main(String[] args){
		double [][]inp = {{17,    24,     1,     8,    15},
				{23,     5,     7,    14,    16},
				{4,     6,    13,    20,    22},
				{10,    12,    19,    21,     3},
				{11,    18,    25,     2,     9}};

		double [][]mask = {{1,     3,     1},
				{0,     5,     0},
				{2,     1,     2}};
		double [][] out = conv(inp,5,5,mask,3,3);
		for(int i=1;i<6;i++){
			for(int j=1;j<6;j++){
				System.out.print(out[i][j]+" ");
			}
			System.out.println();
		}
	}*/
	public static double[][] rotateIt(double [][] arr) { 
		double temp; 
		for (int row1 = 0, row2 = arr.length - 1; row1 <= row2; ++row1, --row2) { 
			for (int col1 = 0, col2 = arr[0].length - 1; col1 < arr[0].length; ++col1, --col2) { 
				if (row1 == row2 && col1 >= col2) { 
					break; 
				} 
				temp = arr[row1][col1]; 
				arr[row1][col1] = arr[row2][col2]; 
				arr[row2][col2] = temp; 
			} 
		}
		return arr;
	}
	public static double[][] conv(double [][]input,int w,int h,double [][]mask,int m_w,int m_h){
		
		mask = rotateIt(mask);
		///Assuming mask is odd sized matrix/////////////
		int p_h = h+m_h - 1;
		int p_w = w+m_w - 1;
		double [][]arr = new double[p_w][p_h];
		double [][]output = new double[p_w][p_h];

		for(int i=0;i<p_w;i++){
			for(int j=0;j<p_h;j++){
				if((i>=m_w/2 && j>=m_h/2) && (i<((m_w/2)+w)) && (j <((m_h/2)+h))){
					arr[i][j] = input[i-m_w/2][j-m_h/2];
					
				}
				else
					arr[i][j] = 0;
			}
		}

		/*for(int i=0;i<p_h;i++){
			for(int j=0;j<p_w;j++){
				System.out.print((int)arr[i][j]+" ");
			}
			System.out.println();
		}
		System.out.println();*/

		for(int i=m_w/2;i<(w+m_w/2);i++){
			for(int j=m_h/2;j<(h+m_h/2);j++){
				double sum = 0;
				for(int k=0;k<m_w;k++){
					for(int l=0;l<m_h;l++){
						sum += arr[i-(m_w/2)+k][j-(m_h/2)+l] * mask[k][l];
					}
				}
				if(sum > 255)
					sum = 255;
				else if(sum <= 0)
					sum = 0;
				output[i][j] = sum;
			}
		}
		return output;
	}

}
