// File: fc_pso_cpp.cpp
// Date: 24/02/2019
// Description: This c++ script implements PSO-based fuzzy controller for boundary-following robot control
//              Please refer to the paper below to see explanation of fuzzy controller parameters.
//    "Juang, C., & Chang, Y. (2011). Evolutionary-Group-Based Particle-Swarm-Optimized 
//    Fuzzy Controller With Application to Mobile-Robot Navigation in Unknown Environments. 
//    IEEE Transactions On Fuzzy Systems, 19(2), 379-392. doi: 10.1109/tfuzz.2011.2104364" 
// 
// Author: Yu-Cheng (Fred) Chang, HDR student, University of Technology Sydney


#include <fstream>
#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <string>
#include <cmath>
#include <time.h>

#include <webots/Robot.h>
#include <webots/distance_sensor.h>
#include <webots/led.h>
#include <webots/motor.h>
#include <webots/robot.h>
#include <webots/lidar.h>
#include <webots/receiver.h>
#include <webots/emitter.h>
#include <webots/inertial_unit.h>
#include <webots/gps.h>

using namespace std;


#define INFMAX    1.0e6
#define INFMIN    1.0e-6
#define _pi       3.1415926
#define min(a,b) (a < b ? a : b)
#define max(a,b) (a > b ? a : b)


// control setting
#define _MaxTimestep 2000
#define _ConstantSpeed 4

// Fuzzy
#define _InVarl 4  // input dimension
#define _OutVarl 1  // output dimension
#define _NumRule 10  // the number of fuzzy rules
#define _CenLimit_max 5  // max center value of a fuzzy set 
#define _CenLimit_min 0  // min           ...  
#define _WidLimit_max 1  // max width value of a fuzzy set 
#define _WidLimit_min 0.3  // min           ...  
#define _ConqLimit_max 3.0 // max consequence value of a fuzzy controller 
#define _ConqLimit_min -3.0 // min           ...  

// Arguments for PSO
#define   _SwarmSize  50     // the size of the population
#define   _MaxIter  200   //  Maximum Number of Iterations
#define   _CT1 1.0         // Personal Learning Coefficient
#define   _CT2 1.0       // Global Learning Coefficient
#define   _IW 0.8         // Inertia Weight
#define   _IW_Damp 0.999   // Inertia Weight Damping Ratio
#define   _PH1 2.05      // constant value for velocity update
#define   _PH2 2.05
#define   _LenChrom  _NumRule*(2*_InVarl+_OutVarl)    // length of an individual

#define _pcross 0.6 // crossover probability
#define _pmutation 0.05 // mutation probability



// --------------------------------------------------------------------------------------
//Classes
class Particle
{
  public:
        float pos[_LenChrom+1],  // position of an individual in PSO
              v[_LenChrom+1];      // velocity of an individual in PSO
        float pbest[_LenChrom+1];  // personal best of an individual
        float fitness,           // fitness value
              pbestfit;          // fitness value of pbest
        
        void update_velocity(void); // velocity update equation
        void update_position(void); // position update equation
        void initpop(void);   // initialise each particle
};


class FuzzyRule
{
 
// center is the centre of Gaussian membership function
// width is the width of Gaussian membership function
// conq is the consequence values of fuzzy rules
  public:
        void memfun(float, int); // fuzzy membership funcion
        int bound[_InVarl+1];  // boundary mark of centre
        float centre[_InVarl+1], 
              width[_InVarl+1],  
              conq[_OutVarl+1]; 
        float mu[_InVarl+1],   // membership values 
              phi;  // firing strength values
       
};



// --------------------------------------------------------------------------------------
// Functions
int robot_send_data(float *, int );
float randn(float, float); 
float Evaluation(Particle );
void InitSwarm(void);
void Report(int );

// Variables
int _timestep; // the time interval between each two control step in simulation
float _w, _wdamp, _c1, _c2; // parameters for PSO
float _posmin[_LenChrom+1], _posmax[_LenChrom+1], // boundaries for position
      _velmin[_LenChrom+1], _velmax[_LenChrom+1]; // boundaries for velocity
      
Particle _pop[_SwarmSize+1],  // Particles
        _gbest; // global best

// Device tag for all devices of the robot in the simulation
static WbDeviceTag left_wheel, right_wheel, tag_emitter_ch01, gps, inertial, lidar_lms291;



// --------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
  int i, j, iter;
  float ph, chi, outbuffer[2]={0.0};
  void InitRobot(void);
  
  srand((unsigned) time(NULL));
  
// setting parameters for PSO
  _w = _IW;
  _wdamp = _IW_Damp;
  _c1 = _CT1;
  _c2 = _CT2;

// If you would like to use Constriction Coefficients for PSO, 
// uncomment the following block and comment the above set of parameters.
// Constriction Coefficients
//-------------------------------------------------------
  ph=_PH1+_PH1;
  chi=2.0/(ph-2+sqrt(ph*ph-4*ph)); // velocity constrain factor
  _w=chi;          // inertia Weight
  _wdamp=1;        // inertia Weight Damping Ratio
  _c1=chi*_PH1;    // personal Learning Coefficient
  _c2=chi*_PH2;    // global Learning Coefficient bug
//-------------------------------------------------------  
  
  wb_robot_init();
  _timestep = wb_robot_get_basic_time_step()*16;
   
  InitRobot();
  
  // Main loop:
  InitSwarm();
  Report(0);

  for(iter=1; iter<=_MaxIter; iter++)
  {
    for(i=1; i<=_SwarmSize; i++)
    {
      _pop[i].update_velocity();
      _pop[i].update_position();
      
      // Evaluation
      _pop[i].fitness = Evaluation(_pop[i]);
            
      // Update Personal Best
      if( _pop[i].fitness < _pop[i].pbestfit )
      {
        _pop[i].pbestfit = _pop[i].fitness;
        for(j=1; j<=_LenChrom; j++)
        {
          _pop[i].pbest[j] = _pop[i].pos[j];
        }
      }
      
      // Update Global Best
      if( _pop[i].fitness < _gbest.fitness )
      {
        _gbest.fitness = _pop[i].fitness;
        for(j=1; j<=_LenChrom; j++)
        {
          _gbest.pos[j] = _pop[i].pos[j];
        }
      }      

    }// loop for particle update ends 
    
    // update inertia weight with damping ratio
    _w =_w*_wdamp; // update inertia weight with damping ratio
    
    // save and report information of gbest
    Report(iter);

// If the robot is capable of following the boundary more than one round, 
// then PSO terminates.    
    if(_gbest.fitness==0)
    {
        iter=_MaxIter+1;
        outbuffer[0]=100.0; // terminating simulation command
        outbuffer[1]=_gbest.fitness; // fitness value
        robot_send_data(outbuffer, 2);
        break;
    }
    //GA insert here
  } // loop for iteration ends

 
  wb_robot_cleanup();


  return 0;
}



// Funcitons for PSO algorithm
// --------------------------------------------------------------------------------------

// Initialization 
void InitSwarm(void)
{
   void AssignConstrain(void);
   int i, j;
   
   AssignConstrain();
   
   _gbest.fitness = INFMAX;
   
   for(i=1; i<=_SwarmSize; i++)
   {
     _pop[i].initpop();
     _pop[i].fitness = Evaluation(_pop[i]);
          
     // initialise personal best
     _pop[i].pbestfit = _pop[i].fitness;
     for(j=1; j<=_LenChrom; j++)
     {
       _pop[i].pbest[j] = _pop[i].pos[j];
     } 
     
     // initialise global best
     if( _pop[i].fitness < _gbest.fitness )
     {
       _gbest.fitness=_pop[i].fitness;
       for(j=1; j<=_LenChrom; j++)
       {
         _gbest.pos[j] = _pop[i].pos[j];
       } 
     }     
   }
}


// Evaluate the performance of a solution (fuzzy controller)
float Evaluation(Particle evapop)
{
   void robot_wait(int );
   int JudgeRobot(float *);
   float *robot_get_lidar_range(WbDeviceTag );   
   float *FuzzyController(float *, FuzzyRule *);
   float co_evaluate(int, float);
   float stability_function(void);
   FuzzyRule *ExtractFuzzy(Particle );
   int step, robot_failure, i;
   float *fout, *obs_dist;
   float cost=0;
   float vr[2+1], in[_InVarl+1], outbuffer[5];
   float obs_area, variance;
   FuzzyRule *rule;
   
   rule = ExtractFuzzy(evapop);
  
   step=0;
   robot_failure=0;
   obs_area=0;
   variance=0;
   // robot control loop
   while((step<=_MaxTimestep) && (robot_failure==0))
   {
     step=step+1;
     // measure area variation to obstacles
     if (step>=2)
     {
        variance+= abs(stability_function()-obs_area);
     }
     // measure distance to obstacles
     obs_dist = robot_get_lidar_range(lidar_lms291);
     // measure area to obstacles
     obs_area = stability_function();
     wb_robot_step(_timestep);

     // input data
     in[1]=obs_dist[1];
     in[2]=obs_dist[2];
     in[3]=obs_dist[3];
     in[4]=obs_dist[4];
     
     // fuzzy controller
     fout=FuzzyController(in , rule);
     
     // calculate left and right wheel speeds.
     // 16.03 is the half distance between the left and right wheels.
     // 9.75 (cm) is the radius of the wheels  
     vr[1] = _ConstantSpeed - (16.3*fout[1])/9.75;
     vr[2] = _ConstantSpeed + (16.3*fout[1])/9.75;
     
     // constrain maximum wheel speed. maximum wheel speed is 5.24 (0.524 m/s).
     for(i=1; i<=2; i++)
     {
       if(vr[i]>5.24)
           vr[i]=5.24;
       else if(vr[i]<-5.24)
           vr[i]=-5.24;
     }
     
     // send command to the robot 
     wb_motor_set_velocity(left_wheel, vr[1]);
     wb_motor_set_velocity(right_wheel, vr[2]);
     
     // measure distance to obstacles
     obs_dist = robot_get_lidar_range(lidar_lms291);
     wb_robot_step(_timestep);
     
     // Judge whether the robot colliding with or moving far away from an  obstacle. 
     // If robot_failure=0, robot is moving in the constrain path.
     //    robot_failure=1, robot is moving far away from an obstacle. 
     //    robot_failure=2, robot is colliding with an obstacle 
     robot_failure = JudgeRobot(obs_dist);

   };
   
   // fitness function
   cost = co_evaluate(step, variance);
   //  reset the environment. 
   outbuffer[0]=robot_failure; // reset command
   outbuffer[1]=cost; // first part of fitness value
   robot_send_data(outbuffer, 2);

   robot_wait(20);

  // robot_send_data(outbuffer, 2);
//  cout << "step_cost = " << cost[0] << ", area variance = " <<cost [1]<< ", final fitness = " <<cost [2]<< endl;
//  cout << "robot_failure = " << robot_failure << endl;
   return cost;
}



// Extract free parameters from a particle
FuzzyRule *ExtractFuzzy(Particle evapop)
{
  static FuzzyRule rule[_NumRule+1];
  float maxcen=INFMIN, mincen=INFMAX;
  int i, j, k, max_id=0, min_id=0;
  int n=2*_InVarl+_OutVarl;

  for(i=1; i<=_NumRule; i++)
  {
    for(j=1; j<=_InVarl; j++)
    {
      rule[i].bound[j]=0;
      rule[i].centre[j]=evapop.pos[(i-1)*n+2*j-1]; // centre of fuzzy set
      rule[i].width[j]=evapop.pos[(i-1)*n+2*j];   // width of fuzzy set

      if(rule[i].centre[j]>maxcen)
      {
        maxcen=rule[i].centre[j];
        max_id=j;
      }

      if(rule[i].centre[j]<mincen)
      {
        mincen=rule[i].centre[j];
        min_id=j;
      }
    }

    rule[i].bound[max_id]=1;
    rule[i].bound[min_id]=-1;


    for(k=1; k<=_OutVarl; k++)
    {
      rule[i].conq[k]=evapop.pos[(i-1)*n+2*_InVarl+k]; // consequence
    }
  }

  return rule;
}



// setting contrain condition
void AssignConstrain(void)
{
  int i,j,k;
  int n=2*_InVarl+_OutVarl;

  // Assign maximum and minimum values for position
  for(i=1; i<=_NumRule; i++)
  {
    for(j=1; j<=_InVarl; j++)
    {
       // constrain condition for the centre of fuzzy sets
       _posmax[(i-1)*n+2*j-1] = _CenLimit_max;
       _posmin[(i-1)*n+2*j-1] = _CenLimit_min;

       // constrain condition for the width of fuzzy sets
       _posmax[(i-1)*n+2*j] = _WidLimit_max;
       _posmin[(i-1)*n+2*j] = _WidLimit_min;
    }

    for(k=1; k<=_OutVarl; k++)
    {
       // constrain condition for the consequence parts
       _posmax[(i-1)*n+2*_InVarl+k] = _ConqLimit_max;
       _posmin[(i-1)*n+2*_InVarl+k] = _ConqLimit_min;
    }
  }

   // Assign maximum and minimum values for velocity
  for(i=1; i<=_LenChrom; i++)
  {
    _velmax[i] = 0.5*(_posmax[i]-_posmin[i]);
    _velmin[i] = -_velmax[i];
  }
}


void Report(int iter)
{
  FILE *ps;
  cout << "Iteration[" << iter << "]: " << "gbest=" << _gbest.fitness << endl;

  // save the so far best fitness velue
  ps=fopen("BestCost__Phase1.dat", "a");
  fprintf(ps, "%f\n", _gbest.fitness);
  fclose(ps);

  // save global best
  ps=fopen("GlobalBest_Phase1.dat", "w");
  for(int i=1; i<=_LenChrom; i++)
  {
    fprintf(ps, "%f\n", _gbest.pos[i]);
  }
  fclose(ps);
}


// random value generator
float randn(float nmin, float nmax)
{

  return ( (float) (rand() / (RAND_MAX + 1.0)) ) * ( nmax - nmin ) + nmin;
}


// velocity update equation
void Particle :: update_velocity(void)
{
  for(int k=1; k<=_LenChrom; k++)
  {
    // velocity update equation
    v[k] = _w*v[k] + _c1*randn(0,1)*( pbest[k] - pos[k] )
                   + _c2*randn(0,1)*( _gbest.pos[k] - pos[k] );

    // Apply Velocity Limits
    v[k]=min(v[k], _velmax[k]);
    v[k]=max(v[k], _velmin[k]);
  }
}



// position update equation
void Particle :: update_position(void)
{
  for(int k=1; k<=_LenChrom; k++)
  {
    pos[k] = v[k] + pos[k];

    // apply position limits and velocity mirror effect
    if(pos[k]>_posmax[k])
    {
      pos[k]=_posmax[k];
      v[k]=-v[k];
    }
    else if(pos[k]<_posmin[k])
    {
      pos[k]=_posmin[k];
      v[k]=-v[k];
    }
  }
}



// Initialise a particle
void Particle :: initpop(void)
{
  for(int i=1; i<=_LenChrom; i++)
  {
    pos[i] = randn(_posmin[i], _posmax[i]);
    v[i] = 0.0;
  }
  fitness = INFMAX;
}



// Functions for fuzzy system
// --------------------------------------------------------------------------------------

// Extract the parameters of fuzzy controller from a particle
float *FuzzyController(float *in, FuzzyRule *rule)
{
    static float fout[_OutVarl+1];
    float num[_OutVarl+1], den[_OutVarl+1];
    int i, j, k;

    for(i=1; i<=_NumRule; i++)
    {
      rule[i].phi=1;
      for(j=1; j<=_InVarl; j++)
      {
         // calculate membership value of a input data point
         rule[i].memfun(in[j], j);

         // calculate firing strength value of a rule (AND operation)
         rule[i].phi=rule[i].phi*rule[i].mu[j];
      }
    }

    // weighted average
    for(k=1; k<=_OutVarl; k++)
    {
      den[k]=0.0;
      num[k]=0.0;
      for(i=1; i<=_NumRule; i++)
      {
        den[k]=den[k]+rule[i].phi;
        num[k]=num[k]+rule[i].phi*rule[i].conq[k];
      }

      if(den[k]<INFMIN)
          den[k]=INFMIN;

      fout[k]=num[k]/den[k];
    }

    return fout;
}



// Fuzzy membership funciton
void FuzzyRule :: memfun(float x, int j)
{
    mu[j] = exp(-0.5*((x-centre[j])*(x-centre[j])/(width[j]*width[j])));

    if(bound[j]==1)
    {
      if(x>centre[j])
          mu[j]=1;
    }
    else if(bound[j]==-1)
    {
       if(x<centre[j])
          mu[j]=1;
    }
}



// Functions for robot control
// --------------------------------------------------------------------------------------

// Initialize all devices of the robot
void InitRobot(void)
{
    left_wheel  = wb_robot_get_device("left_wheel");
    right_wheel = wb_robot_get_device("right_wheel");
    tag_emitter_ch01 = wb_robot_get_device("emitter_ch01");
    gps = wb_robot_get_device("pioneer_gps");
    inertial = wb_robot_get_device("pioneer_inertial");
    lidar_lms291 = wb_robot_get_device("lms291");

    wb_gps_enable(gps,_timestep/2);
    wb_inertial_unit_enable(inertial,_timestep/2);
    wb_lidar_enable(lidar_lms291,_timestep/2);


    // Enable wheels rolling
    wb_motor_set_position(left_wheel, INFINITY);
    wb_motor_set_position(right_wheel, INFINITY);
    wb_motor_set_velocity(left_wheel, 0.0);
    wb_motor_set_velocity(right_wheel, 0.0);

    wb_robot_step(_timestep);
}


// Send data to simulation environment supervisor
int robot_send_data(float *a, int ldata)
{
   int buffersize=(ldata+1)*sizeof(float);
   int sent=0;
   float *outbuffer;

   outbuffer = new float [ldata+1];

   outbuffer[0]=ldata;
   for(int i=0; i<(int)ldata; i++)
   {
     outbuffer[i+1]=a[i];
   }

   sent=wb_emitter_send(tag_emitter_ch01, outbuffer, buffersize);

   delete [] outbuffer;

   wb_robot_step(_timestep);

   return sent;
}



// get the distance values to the obstacles from lidar information
float *robot_get_lidar_range(WbDeviceTag lidar_tag)
{
  int i, j;
  int area[8][2] = {{0,20}, {20,50}, {50,70}, {70,90}, // left scanning area
                    {90,110}, {110,130}, {130,160}, {160,179}};  // Right scanning area
  static float obs_dist[8+1]={0.0};
  const float *lidar_values = wb_lidar_get_range_image(lidar_tag);

  for(i=0; i<8; i++)
  {
    obs_dist[i+1]=lidar_values[area[i][0]];
    for(j=area[i][0]; j<=area[i][1]-1; j++)
    {
        obs_dist[i+1] =  min(obs_dist[i+1], lidar_values[j]);
    }

  }
  return obs_dist;
}

// get the area values to the obstacles from lidar information
float robot_get_obs_area(WbDeviceTag lidar_tag)
{
  int i, j;
  int area[8][2] = {{0,20}, {20,50}, {50,70}, {70,90}, // left scanning area
                    {90,110}, {110,130}, {130,160}, {160,179}};  // Right scanning area
  float obs_area=0.0;
  const float *lidar_values = wb_lidar_get_range_image(lidar_tag);

  for(i=0; i<4; i++)
  {
    for(j=area[i][0]; j<=area[i][1]-1; j++)
    {
        obs_area+=_pi*lidar_values[j]*lidar_values[j]/360; //calculate
    }

  }
  return obs_area;
}




// Judge whether the robot colliding with or moving far away from an obstacle.
int JudgeRobot(float *dist)
{
   int i, robot_failure;
   float min_dist=INFMAX;

   robot_failure=0;

   for(i=1; i<=8; i++)
   {
     min_dist=min(min_dist, dist[i]);
   }

   if(min_dist<0.5)   // define collision condition
      robot_failure=2;

   if(dist[1]>3.5) // define far-away condition
      robot_failure=1;


   return robot_failure;
}


// Robot delay function
void robot_wait(int waitime)
{
    for(int i=1; i<=waitime; i++)
    {
       wb_motor_set_velocity(left_wheel, 0);
       wb_motor_set_velocity(right_wheel, 0);
       wb_robot_step(_timestep);
    }
}

//evaluate distence
void distence_function()
{

}

//evaluate stability
float stability_function(void)
{
    float _obs_area;
//    obs_area = robot_get_obs_area(lidar_lms291);
    _obs_area=  wb_lidar_get_range_image(lidar_lms291)[0];
    return _obs_area;
}
//evaluate speed
// float speed_function()
// {

// }

//evaluate speed
// float interpretability_function()
// {

// }
//calculate final fitness
float co_evaluate(int _step, float _variance)
{
    //define 3 differennt score for calculation the final fitness
    float time_score, stable_score, final_score;
    float cStep = (float)_step;
    time_score=1/cStep;
    //if the robot move one step more, its stability can be ignored
    stable_score=((2/(1+exp(-_variance/cStep)))-1)*( (1/cStep) - (1/(cStep+1)));
    final_score=time_score+stable_score;

    if (_step>=_MaxTimestep)
    {
        return 0;
    }
    else
    {
        return final_score;
    }
}

void GA_transformation(int Ng, *Particle group, int t)
{
    int i,j,g_index[Ng/2+1],p_index[2+1],len;
//    sort particles in the group with performance;
    sort_particle(group);
//    crossover the best particle with others;
//select Ng /2 groups
    g_index = select_index(Ng/2, Ng);
    for(i=1;i<=Ng/2;i++)
    {
        len=group[g_index[i]-1].length-1;
        if(g_index[i]!=Ng)
        {   //crossover 2 particles within the group for all groups
            p_index = select_index(2, len); // for each group, select 2 parents from all particles in the group
            pop_crossover(group[g_index[i]][p_index[1]],group[g_index[i]][p_index[2]]);
        }
        else
        {   //crossover all particles for the worst group
            p_index = select_index(len, len);
            pop_crossover(group[g_index[i]][p_index[1]],group[g_index[i]][p_index[2]]);
            if (len%2!=0)
            { //If Nworst is odd,crossover the worst the worst particle in the group with the second-worst performance
                pop_crossover(group[g_index[i]-1][group[g_index[i]-1].length-1], group[g_index[i]][p_index[1]]);
                for (j=2;j<len;j+=2)
                {
                    pop_crossover(group[g_index[i]][p_index[j]],group[g_index[i]][p_index[j+1]]);
                }
            }
            else
            {
                for (j=1;j<len;j+=2)
                {
                    pop_crossover(group[g_index[i]][p_index[j]],group[g_index[i]][p_index[j+1]]);
                }
            }


        }
    }

//mutation (only mutate velocity)
    for (i=1,i<=_SwarmSize,i++)
    {
        if(randn(0,1)<=_pmutation)
        {
            pop_mutation(pop[i],t);
        }
    }

}





void sort_particle(*Particle[] p,int n)
{
      int left,right;
            float num;
            Particle *temp;
      int middle,j,i;
      for(i = 2;i <= n;i++)
      {
          left = 1;// 准备
          right = i-1;
                    temp=p[i]
          num = p[i].fitness;         
          while( right >= left)// 二分法查找插入位置
          {
              middle = ( left + right ) / 2; //　指向已排序好的中间位置
              if( num < p[middle].fitness )// 即将插入的元素应当在在左区间
               right = middle-1;
                            else                    //　即将插入的元素应当在右区间
                       left = middle+1;    
          }
//每次查找完毕后，left总比right大一，a[left]总是存放第一个比num大的数，因此应从此处开始，每            //个元素右移一位，并将num存入a[left]中，这样就保证了a[0...i]是排好序的
          for( j = i-1;j >= left;j-- )//　后移排序码大于R[i]的记录
              p[j+1] = p[j];
          p[left] = temp;// 插入
      }
}
//crossover function
void pop_crossover(*Particle p1, *Particle p2)
{
    int site[2+1]=select_index(2,_LenChrom);
    float temp1[_LenChrom],temp2[_LenChrom];
    float gamma1=randn(0,1),gamma2=randn(0,1),gamma3=randn(0,1);
    float gamma4=randn(0,1),gamma5=randn(0,1),gamma6=randn(0,1);
    int i，group_id1，group_id2;
//    create parent1 and parent2
    for (i=1;i<=_LenChrom;i++)
    {
        parent1[i]=p1.pos[i];
        parent2[i]=p2.pos[i];
    }
    //crossover parameters
    for (i=1;i<=site[1];i++)
    {
        p1.pos[i]=gamma1*parent1[i]+(1-gamma1)*parent2[i];
        p2.pos[i]=gamma1*parent2[i]+(1-gamma1)*parent1[i];
    }
    for (i=site[1]+1;i<=site[2];i++)
    {
        p1.pos[i]=gamma2*parent1[i]+(1-gamma2)*parent2[i];
        p2.pos[i]=gamma2*parent2[i]+(1-gamma2)*parent1[i];
    }
    for (i=site[2]+1;i<=_LenChrom;i++)
    {
        p1.pos[i]=gamma3*parent1[i]+(1-gamma3)*parent2[i];
        p2.pos[i]=gamma3*parent2[i]+(1-gamma3)*parent1[i];
    }

//    insert into most similer group
    group_id1=insert_particle(p1);
    group_id2=insert_particle(p2);

    //        update velocity
    for (i=1;i<=_LenChrom;i++)
    {
        p1.v[i]=gamma4*(p_group[group_id1][1].pos[i]-p1.pos[i])+gamma5*(parent1[i]-p1.pos[i])+gamma6*(parent2[i]-p1.pos[i]);
        p2.v[i]=gamma4*(p_group[group_id1][1].pos[i]-p2.pos[i])+gamma5*(parent1[i]-p2.pos[i])+gamma6*(parent2[i]-p2.pos[i]);
    }


}
//select m numbers of indeces from total length of m
int *select_index(int m, int n)
{
    int i,j,index[m], a[n]={0};
    for(i=1;i<=m;)
    {
        j=rand()%n+1;        //chose a value from n
        if(a[j]==0) //if the index is selected? if not record this index
        {
            index[i]=j;
            a[j]=1;
            i++;
        }
    }
    return index
}

int insert_particle (*Particle p)
{
    int group_id;
    float similarity;
    similarity=calculate_similarity(p_group[1][1],p);
    for(i=1;i<=p_goup.length,i++)
    {
        if (similarity>=calculate_similarity(p_group[i][1],p))
        {
            group_id=i;
            similarity=calculate_similarity(p_group[i][1],p);
        }
    }
//    插入到组里
    return group_id;
}

//mutation function
void pop_mutation(*Particle p, int t)
{
    int i,j;
    for(i=1;i<=_LenChrom;i++)
    {
        float sman=smin=_pop[j].pos[i];
        float delta,randv;
        float gamma7=randn(0,1),gamma8=randn(0,1);
//find max and min position values for index [i]
        for(j=i;j<=_SwarmSize;j++)
        {
            smax=max(smax,_pop[j].pos[i]);
            smin=min(smax,_pop[j].pos[i]);
        }
        delta=smax-smin;
        randv=max(exp(-(t*t/_MaxIter/_MaxIter)),0.1)*0.8*(gamma8*2*delta-delta);
        p.v[i]=(1-gamma7)p.v[i]+gamma7*randv;
    }
}