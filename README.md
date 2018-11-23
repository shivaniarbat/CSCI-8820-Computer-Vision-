# CSCI-8820-Computer-Vision
1. Iterative Connected Component Labelling
2. Distance Transform
3. Thresholding 

* Threshold using peakiness detection : Evaluation Criteria used to select the pair of grayscale peaks and the intervening valleys – 
i.	Peaks should be dominant. (set of peaks selected must have count of occurrences of particular gray value greater than or equal to 2500) (pi ,….., pj)
ii.	pi and pj should not be same.
iii.	Peaks selected should be well separated at least by value of 70.
iv.	Valleys should be deep enough. (set of peaks selected must have count of occurrences of gray values smaller than or equal to 300) (vij )
v.	Valleys should be near to center of both the peaks. (in the images tested the offset of +-20 is used to accommodate more valleys, as valleys exact at center were very few and to calculate measure of goodness only 1 or zero combination of peaks and valleys were found. Especially for test3.img image none of the valleys for this criteria falls into this category and thus no output is generated and image is not segmented at all)
vi.	Measure of Goodness (MOG) is calculated as follows: - 
MOG = (|pi – pj| * (H(pi) + H(pj))/2) / vij
	Where, 
		pi      – peak from selected dominant set of peaks 
		pj.        – peak from selected dominant set of peaks 
		vij.       – valley from selected valleys satisfying above criteria 
		H(pi) – count of pixels that have gray values pi
		H(pj) – count of pixels that have gray values pj

* Dual Thresholding: Histograms obtained are not bimodal histograms. Conservative thresholds are selected. T1 - mean of entire array and T2 - highest peak from pixels having gray values above 128. 
Selection mean for threshold T1 which is neither peak or valley which give us resulting region 1 which we may say are core for some of the objects. 
For T2 -  selecting T2 from pixels values greater than 128 give us buffer region which have large set of pixels to determine whether they are connected to region 1 or belongs to region 3. Even if the peak is at 255, pixels which are not connected to region 1 pixels we can determine the foreground and background in the image properly. Results obtained for test 1 and test 2 have T2 at 255 and T1 is mean.

4. Edge Detection
