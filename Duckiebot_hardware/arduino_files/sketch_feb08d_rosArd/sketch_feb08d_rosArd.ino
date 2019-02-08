#include <ros.h>
#include <geometry_msgs/Twist.h>

ros::NodeHandle  nh;

/////important parameters
int motor_l0 = 10;
int motor_l1 = 11;
int motor_r0 = 5;
int motor_r1 = 6;

//%gain calculated to 
//%gain = 0.644;
float gain = 0.644;
float trim = 0;

//Measured Parameters
float Pi = 3.14 ; 
float L = 0.09 ;  //%(in m)
float R = 0.041 ;
float D = 2* R;

int rpm_limit = 100;//means at v = 1 motor will run at 100rpm and not 150 rpm
int limit = rpm_limit * 1.7; //analog limit corresponds to 100 rpm (1.7 =(255/150)--> 150rpm comes from 255 analog value)

float Wr,Wl,Nr,Nl,V,W,Fr_limited,Fl_limited,Fr,Fl;

 
void messageCb(const geometry_msgs::Twist& msg)
{
   V = msg.linear.x;
   W = msg.angular.z;
  
  
  if(V < -1 || V >1)
    Serial.println( "input Velocity between -1 and 1" ); 
   
   //gain and trim parameters to be tuned for dc motors;
   Wr = ((gain + trim) * (V + 0.5 * W * L))/ R ;  //%angular velocity right(rad/s);
   Wl = ((gain - trim) * (V - 0.5 * W * L))/ R ;  //%angular velocity left(rad/s);
   
   //converting from rad/s to rpm
   Nr = Wr * 9.5493;
   Nl = Wl * 9.5493;
   
   //limiting rpm to 100 
   Fr = (Nr/100.0)*limit;
   Fl = (Nl/100.0)*limit;
   
   //limiting voltage to 170 even if Fr goes above 100 corresponding to 100 rpm
   int Fr_limited = max(min(Fr, limit), -limit);
   int Fl_limited = max(min(Fl, limit), -limit);
   
   //if velocity neg reverse the direction;
   if(V < 0){
     analogWrite(motor_l0,Fl_limited);
     analogWrite(motor_r0,Fr_limited);
     analogWrite(motor_l1,0);
     analogWrite(motor_r1,0);
    
   }else{    
    analogWrite(motor_l1,Fl_limited);
    analogWrite(motor_r1,Fr_limited);
    analogWrite(motor_l0,0);
    analogWrite(motor_r0,0); }
   
}

//subscriber topic 
ros::Subscriber<geometry_msgs::Twist> sub("cmd_vel", &messageCb );

void setup()
{
  //sets every pin to output
  pinMode(motor_l0, OUTPUT);
  pinMode(motor_l1, OUTPUT);
  pinMode(motor_r0, OUTPUT);
  pinMode(motor_r1, OUTPUT);
    
  nh.initNode();
  nh.subscribe(sub);
    
}

void loop()
{
  nh.spinOnce();
  delay(1);
}
