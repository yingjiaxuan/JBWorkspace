//
//  main.c
//  ObjCTut
//
//  Created by 降龙大嘴 on 2020/11/4.
//

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <bool.h>

int main(int argc, const char * argv[]) {

    float fTemp;
    printf("Enter Temp in F : ");
    //scanf("%f", &fTemp);
    fTemp = 70;

    float cTemp = (fTemp - 32)/1.8;

    printf("%.1f degrees Celsius\n", cTemp);

    int bigInt = 2147483647;
    printf ("Big Int : %d\n", bigInt +1);

    printf ("Min Float : %e\n", FLT_MIN);
    printf ("Max Float : %e\n", FLT_MAX);

//    short %d
//    int %double
//    long %ld
//    float %f
//    double %lf
//    char %c
    //
    return 0;
}