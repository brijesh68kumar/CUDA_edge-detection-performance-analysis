import java.awt.Color;
import java.awt.Point;
import java.awt.image.BufferedImage;
import java.awt.image.PixelInterleavedSampleModel;
import java.awt.image.Raster;
import java.awt.image.SampleModel;
import java.awt.image.WritableRaster;
import java.io.*;
import java.net.URL;
import java.io.File;
import javax.imageio.ImageIO;
import java.util.Date;

public class LoG {
	public static void main(String [] args){
		SampleModel sample = null;
		int mask_size = 5;
		double sigma = 1.2;
		double [][]mask = new double[mask_size][mask_size];
		double x,y;
		for(int i=0;i<mask_size;i++){
			for(int j=0;j<mask_size;j++){
				x = i-(mask_size/2);
				y = j-(mask_size/2);
				mask[i][j] = -1.0 * ((2.0 * sigma * sigma - (x*x+y*y))/(2.0 * Math.PI* sigma * sigma * sigma * sigma)) * Math.exp(-(x*x+y*y)/(2.0 *sigma * sigma));
				//mask[i][j] = (-1.0/(Math.PI*sigma*sigma*sigma*sigma)) *(1 - ((x*x+y*y)/(2.0*sigma * sigma))) * Math.exp(-(x*x+y*y)/(2.0 *sigma * sigma));
			}
		}
		//print
		/*for(int i=0;i<mask_size;i++){
			for(int j=0;j<mask_size;j++){
				System.out.printf("%.1f ",mask[i][j]);	
			}
			System.out.println();
		}*/
		///////////read image////////////////////
		int w = 0,h = 0;
		double pixels[][]=null;
		try 
		{
                        
			BufferedImage img = ImageIO.read(new File(System.getProperty("user.dir")+"/gray_squirrel.jpg"));
			Raster raster = img.getData();
			w=raster.getWidth();h=raster.getHeight();
			//System.out.println(w+" "+h);
			sample = raster.getSampleModel();

			pixels = new double[w][h];
			for (int i=0;i<w;i++)
			{
				for(int j=0;j<h;j++)
				{
					pixels[i][j]= raster.getSample(i,j,0);//img.getRGB(i, j);//
				}
			}
			
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}

		///////////////////////////convolution/////////////////////
		Convolution cn = new Convolution();
		//double [][]pixels2 = cn.convolution2DPadded(pixels, w, h, mask, mask_size, mask_size);
		
		double [][]log_pixels = cn.conv(pixels, w, h, mask, mask_size, mask_size);
		
		double [][]useful_pixels = new double[w][h];
		for(int i= 0;i< w;i++){
			for(int j= 0;j < h;j++){
				useful_pixels[i][j] = log_pixels[i+mask_size/2][j+mask_size/2];
			}
		}
		
		double [][]zero_corr_mask = {{0,0,-1 ,0 ,0}, {0,-1,-2,-1,0}, {-1,-2,16,-2,-1}, {0,-1,-2,-1,0}, {0,0,-1,0,0}};
		
		double [][]zero_corr_out = cn.conv(useful_pixels, w, h, zero_corr_mask, 5, 5);
		
		double [][]final_output = new double[w][h];
		for(int i= 0;i< w;i++){
			for(int j= 0;j < h;j++){
				final_output[i][j] = zero_corr_out[i+2][j+2];
			}
		}
		
		/*for(int i=0;i<10;i++){
			for(int j=0;j<10;j++){
				System.out.printf("%.1f ",pixels2[i][j]);
			}
			System.out.println();
		}*/
		///conv end//////////
		
		
		///////////////////////////////WRITE image///////////////////////////////////

		WritableRaster w_raster= Raster.createWritableRaster(sample, new Point(0,0));
		//WritableRaster w_raster= Raster.createWritableRaster(new PixelInterleavedSampleModel(0, w, h, 1, 1920, new int[] {0}), new Point(0,0));

		for(int i=0;i<w;i++)
		{
			for(int j=0;j<h;j++)
			{
				w_raster.setSample(i,j,0,final_output[i][j]);

			}
		}
		BufferedImage image = new BufferedImage(w,h,BufferedImage.TYPE_BYTE_GRAY); 
		image.setData(w_raster);
		try{
			File f = new File(System.getProperty("user.dir")+"/pic_final.jpg");
			ImageIO.write(image, "jpg",f );

		}catch(Exception e){
			e.printStackTrace();
		}
	}
}
