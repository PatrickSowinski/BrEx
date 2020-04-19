import { Component, OnInit, AfterViewInit } from '@angular/core';

import { VideoService } from '../services/video.service';

@Component({
  selector: 'app-exercise',
  templateUrl: './exercise.component.html',
  styleUrls: ['./exercise.component.css']
})
export class ExerciseComponent implements OnInit, AfterViewInit {

  baseUrl = 'http://localhost:5000/demo_mode'
  public cameraAllowed = false;

  public earlyInstructions = [
    'Please stand up and turn sideways',
    'Adjust your shoulder and your hip to the camera',
    'Lets start!',
  ];

  public instructions = [
    { value: 'Inhale from your belly and hold your breath for 5 seconds', action: 'belly_inhale'},
    { value: 'Exhale', action: 'belly_exhale' },
    { value: 'Inhale from your belly and hold your breath for 5 seconds', action: 'belly_inhale'},
    { value: 'Exhale', action: 'belly_exhale' },
    { value: 'Inhale from your lung and hold your breath for 5 seconds', action: 'lung_inhale'},
    { value: 'Exhale', action: 'lung_exhale' },
    { value: 'Inhale from your lung and hold your breath for 5 seconds', action: 'lung_inhale'},
    { value: 'Exhale', action: 'lung_exhale' },
    // 'Inhale and hold your breath for 5 seconds',
    // 'Exhale',
  ];

  public index = 0;
  public earlyIndex = 0;

  // buttons
  public displayStartButton = false;
  public displayNextButton = true;
  public displayResultButton = false;
  public enableStart = false;

  // result calculation
  public resultCalc: boolean;
  public resultFound = false;
  public instructionsFinished = false;

  constructor(private videoService: VideoService) { }

  ngOnInit(): void {
  }

  ngAfterViewInit(): void {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
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
      this.calculateBreathe(this.index);
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
    console.log("bittiii")
    this.displayResultButton = true;
    this.instructionsFinished = true;
    this.resultFound = false;
  }

  nextClick() {
    this.earlyIndex += 1;
    if (this.earlyIndex >= 2) {
      this.displayNextButton = false;
      this.getSuccess();
      this.displayStartButton = true;
    }
  }

  calculateBreathe(index) {
    this.videoService.getCalculations().subscribe(data => {
      const calcs = data.calculations;
      this.resultFound = true;
      if (this.instructions[index].action === calcs) {
        this.resultCalc = true;
      } else {
        this.resultCalc = false;
      }
    });
  }

  getSuccess() {
    this.videoService.getStart().subscribe(data => {
      const start = data;
      if (start.success) {
        this.enableStart = true;
      }
    });
  }

}
