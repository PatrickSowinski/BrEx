import { Component, OnInit, AfterViewInit } from '@angular/core';

@Component({
  selector: 'app-exercise',
  templateUrl: './exercise.component.html',
  styleUrls: ['./exercise.component.css']
})
export class ExerciseComponent implements OnInit, AfterViewInit {

  baseUrl = 'http://localhost:5000/video_feed'
  public cameraAllowed = false;

  public earlyInstructions = [
    'Please stand up and turn sideways',
    'Adjust your shoulder and your hip to the camera',
    'Lets start!',
  ];

  public instructions = [
    'Inhale and hold your breath for 5 seconds',
    'Exhale',
    'Inhale and hold your breath for 5 seconds',
    'Exhale',
    'Inhale and hold your breath for 5 seconds',
    'Exhale',
    'Inhale and hold your breath for 5 seconds',
    'Exhale',
    'Inhale and hold your breath for 5 seconds',
    'Exhale',
    'Do a big cough'
  ];

  public index = 0;
  public earlyIndex = 0;

  // buttons
  public displayStartButton = false;
  public displayNextButton = true;
  public displayResultButton = false;

  constructor() { }

  ngOnInit(): void {
  }

  ngAfterViewInit(): void {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        // this.video.nativeElement.srcObject = stream;
        // this.video.nativeElement.play();
        // this.postStream(this.video.nativeElement.srcObject);
        this.cameraAllowed = true;
      });
    }
  }

  startExercise() {
    this.displayStartButton = false;
    let timeRange;
    if (this.index % 2 == 0) {
      timeRange = 7;
    } else {
      timeRange = 3;
    }
    this.startTimer(timeRange).then(() => {
      this.index++;
      if (this.index <= this.instructions.length) {
        this.startExercise();
      } else {
        this.stopExercise();
      }
    });


  }

  startTimer(timeleft) {
    return new Promise((resolve, reject) => {
      const countdownTimer = setInterval(() => {
        timeleft--;

        document.getElementById('countdown').innerHTML = timeleft;

        if (timeleft <= 0) {
          document.getElementById('countdown').innerHTML = '';
          clearInterval(countdownTimer);
          resolve(true);
        }
      }, 1000);
    });
  }

  stopExercise() {
    this.displayResultButton = true;
  }

  nextClick() {
    this.earlyIndex += 1;
    if (this.earlyIndex >= 2) {
      this.displayNextButton = false;
      this.displayStartButton = true;
    }
  }

}
