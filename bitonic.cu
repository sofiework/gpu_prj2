/**
 * 
 * The student is required to add content to this file.  This file is
 * your implementation of the project and will be submitted for grading.
 * 
 */



#include "main.h"
#include "student.h"



/**********************************************************************************
 * 
 * Implement your GPU device kernel(s) here (e.g., the bitonic sort kernel).
 * 
 **********************************************************************************/

 


/**********************************************************************************
 * 
 * Implement your utility functions here
 * 
 **********************************************************************************/




/**********************************************************************************
 * 
 * Implement the three main program functions
 * 
 **********************************************************************************/



/**
 * This function transfers data from Host to Device
 */
void host_to_dev()
{

 
}

/**
 * This function performs the bitonic sort and merge by calling the
 * kernels you have defined in the section above
 */
void bitonic_sort()
{

 
}

/**
 * This functiuon transfers the sorted data from Device to Host
 */
DTYPE *dev_to_host()
{

    // You may modify the return value as long as it is a DTYPE pointer to the 
    // GPU sorted array and is the correct size
    return arrSortedGpu;
}
