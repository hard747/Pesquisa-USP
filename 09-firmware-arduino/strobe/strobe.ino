#include <SPI.h>
#include <avr/io.h>

//#define ALOW_DELAY_AFTER_EXTERNAL_INTERRUPT
//#define OUTPUT_SYNC_COMPARE
#define N_PULSES_FROM_SAME_STROBE

int const camera_interrupt_pin = 3; // interrupt 0
int const camera_interrupt_compare_pin = 2; // interrupt 1

// time in microseconds on timer 1
long syncbar_onperiod = 32000L;


//-----------SYNCLED-----------
// led used to generate bright and dark pupil

volatile int syncled_state = 0;
#ifdef N_PULSES_FROM_SAME_STROBE
volatile int changeleds = 0;
volatile int NPULSES = 1;
volatile int pulsecount = 0;
#endif
volatile bool donothing = false;
volatile unsigned long ledonperiod = 80UL;
volatile unsigned long leddelayon = 192UL;

volatile unsigned long overtakes = 0UL;

volatile unsigned long extcounter;
volatile unsigned long extovertakes;
volatile bool newexternalclock = false;

volatile unsigned long cameracounter;
volatile unsigned long cameraovertakes;
volatile bool newcameraframe = false;

volatile bool turnoff_timer3 = false;


volatile int WHATISHIGH = LOW;
volatile int WHATISLOW = HIGH;

int const calibration_leds_pin = 23; //A5
int const ir_led_pin = 22; //A4

//----------SYNCLED------------


// the setup routine runs once when you press reset:
void setup() {  
  
  //----------- TIMER 1 SETUP -----------
  cli(); //stop interrupts
  TCCR1A = 0;// set entire TCCR1A register to 0
  TCCR1B = 0;// same for TCCR1B
  TCNT1  = 0;//initialize counter value to 0
  // set compare match register for syncbar_onperiod
  //OCR1A = (16000000/((1000000/syncbar_onperiod)*8))-1; // = [ (16*10^6) / ((1000000/syncbar_onperiod)*prescalervalue) ] - 1 (must be <65536)
  unsigned long tmp = 2UL*syncbar_onperiod - 1;
  OCR1A = tmp;
  
  // turn on CTC mode
  TCCR1B |= (1 << WGM12);
  // Set CS11 bits for 8 prescaler
  TCCR1B |= (1 << CS11);
  // enable timer1 compare interrupt
  TIMSK1 |= (1 << OCIE1A);
  //----------- TIMER 1 SETUP -----------
  
  //----------- TIMER 3 SETUP -----------
  cli(); //stop interrupts
  TCCR3A = 0;// set entire TCCR3A register to 0
  TCCR3B = 0;// same for TCCR3B
  
  // turn on CTC mode
  TCCR3B |= (1 << WGM32);
  // Set CS32 bits for 256 prescaler
  TCCR3B |= /*(1 << CS31) | */ (1 << CS32);
  // disable timer compare interrupt
  TIMSK3 &= !(1 << OCIE3A);
  
  sei();//allow interrupts
  //----------- TIMER 3 SETUP -----------
  

  
  pinMode(camera_interrupt_pin,INPUT);
  digitalWrite(camera_interrupt_pin,LOW);
  // pin 3 interrupt
  attachInterrupt(0, int0, RISING);
  pinMode(camera_interrupt_compare_pin,INPUT);
  digitalWrite(camera_interrupt_compare_pin,LOW);
  // pin 2 interrupt
  attachInterrupt(1, int1, RISING);

  pinMode(calibration_leds_pin,OUTPUT);
  pinMode(ir_led_pin,OUTPUT);
  digitalWrite(calibration_leds_pin,WHATISHIGH);
  digitalWrite(ir_led_pin,WHATISLOW);
  syncled_state = 0;
  
  // initialize serial communication:
  Serial.begin(115200);
  while (!Serial); // wait for Leonardo enumeration, others continue immediately

}


void loop() {
#ifdef OUTPUT_SYNC_COMPARE
  if (newcameraframe) {
    // print time at the interrupt
    Serial.print("1 ");
    Serial.print(cameraovertakes,DEC);
    Serial.print(" ");
    Serial.println(cameracounter,DEC);
    newcameraframe = false;
  }
  if (newexternalclock) {
    // print time at external clock interrupt 
    Serial.print("2 ");
    Serial.print(extovertakes,DEC);
    Serial.print(" ");
    Serial.println(extcounter,DEC);
    newexternalclock = false;
  }
#endif

  // read serial:
  if (Serial.available() > 0) {
    byte inByte = Serial.read();
    if (inByte == 'p') {
      long tmp = Serial.parseInt();
      if (tmp > 100 && tmp <= 32600) {
        cli(); //stop interrupts
        TCNT1  = 0;//initialize counter value to 0
        //OCR1A = (16000000/((1000000/ledonperiod)*8))-1;
        tmp = 2*tmp - 1;
        OCR1A = tmp;
        sei(); //enable interrupts
        Serial.println(OCR1A,DEC);
      }
    }
    else if (inByte == 'h') {
      long tmp = Serial.parseInt();
      if (tmp > 100) {
        Serial.println(tmp,DEC);
        leddelayon = tmp;
      }
    }
    else if (inByte == 'k') {
      long tmp = Serial.parseInt();
      if (tmp >= 80) {
        Serial.println(tmp,DEC);
        ledonperiod = tmp;
      }
    }
    else if (inByte == 'V') {
      Serial.println("STROBE:");
    }
    else if (inByte == 's') {
      donothing = true;
    }
  }
}

void setLeds(volatile int &state)
{
  if (state == 0) 
  {
#ifdef N_PULSES_FROM_SAME_STROBE
    if (changeleds==0) {
#endif
      digitalWrite(calibration_leds_pin,WHATISHIGH);
      digitalWrite(ir_led_pin,WHATISLOW);
#ifdef N_PULSES_FROM_SAME_STROBE
    } else if (changeleds==1) {
      digitalWrite(ir_led_pin,WHATISHIGH);
      digitalWrite(calibration_leds_pin,WHATISLOW);
    }
#endif
    state++;
  }
  else if (state == 2) 
  {
#ifdef N_PULSES_FROM_SAME_STROBE
    if (changeleds==1) {
#endif
      digitalWrite(ir_led_pin,WHATISHIGH);
      digitalWrite(calibration_leds_pin,WHATISLOW);
#ifdef N_PULSES_FROM_SAME_STROBE
    } else if (changeleds==0) {
      digitalWrite(calibration_leds_pin,WHATISHIGH);
      digitalWrite(ir_led_pin,WHATISLOW);
    }
#endif
    state++;
  }
  else if (state == 1 || state == 3)
  {
    digitalWrite(ir_led_pin,WHATISHIGH);
    digitalWrite(calibration_leds_pin,WHATISHIGH);
  }
}

// external clock interrupts
void int0()
{
  // save
  // save
  extcounter = TCNT1;
  extovertakes = overtakes;
  newexternalclock = true;
  
  if (donothing) {
    donothing = false;
    return;
  }
  
  syncled_state++;
#ifdef N_PULSES_FROM_SAME_STROBE
  if (syncled_state == 2) {
    pulsecount++;
    if (pulsecount >= NPULSES) {
      pulsecount=0;
      changeleds++;
    }
    if (changeleds>1)
      changeleds = 0;
  }
#endif
  if (syncled_state > 3) {
    syncled_state = 0;
#ifdef N_PULSES_FROM_SAME_STROBE
    pulsecount++;
    if (pulsecount >= NPULSES) {
      pulsecount=0;
      changeleds++;
    }
    if (changeleds>1)
      changeleds = 0;
#endif
  }
    
  if (syncled_state == 0 || syncled_state == 2) {
#ifdef ALOW_DELAY_AFTER_EXTERNAL_INTERRUPT
    TCNT3  = 0;//initialize counter value to 0
    long tmp = leddelayon/16L - 1;
    OCR3A = tmp;
    TIMSK3 |= (1 << OCIE3A); // enable timer compare interrupt
#else
    TCNT3  = 0;//initialize counter value to 0
    long tmp = ledonperiod/16L - 1;
    OCR3A = tmp;
    TIMSK3 |= (1 << OCIE3A); // enable timer compare interrupt
    //donothing = true; //set this only if the onperiod is greater than frame period
    setLeds(syncled_state);
#endif
  }
  
}

// real camera interrupts
void int1()
{  
  // save
  cameracounter = TCNT1;
  cameraovertakes = overtakes;
  newcameraframe = true;
}

ISR(TIMER1_COMPA_vect){
  overtakes++;
}

ISR(TIMER3_COMPA_vect)
{
#ifdef ALOW_DELAY_AFTER_EXTERNAL_INTERRUPT 
  if (syncled_state == 0 || syncled_state == 2) 
  {
    TCNT3  = 0;//initialize counter value to 0
    long tmp = ledonperiod/16L - 1;
    OCR3A = tmp;
    //donothing = true; //set this only if the onperiod is greater than frame period
  }
#endif
  setLeds(syncled_state);
  TCNT3  = 0;//initialize counter value to 0
  OCR3A = 0xFFFF;
}

