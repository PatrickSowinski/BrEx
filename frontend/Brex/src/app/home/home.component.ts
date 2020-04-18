import {Component, ElementRef, OnInit, ViewChild, AfterViewInit} from '@angular/core';
import {VideoService} from '../services/video.service';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent implements OnInit, AfterViewInit {

  baseUrl = 'http://localhost:5000/video_feed'
  cameraAllowed = false;
  displayCamera = false;

  instructions = [
      'Press the button to start',
      'Take 5 deep breaths in and each time hold your breath for 5 seconds',
      'Do a big cough',
      'Take 5 deep breaths in and each time hold your breath for 5 seconds',
      'Do a big cough'
  ]
  index = 0;

  motto = "Breath correct to survive!"
  explanation = "The virus infects the lung cells and an excess of inflammation liquid comes out. this inflammation liquid prevents O2 transfer to blood cells and patient starts to have breathing problems. the current treatment is to perform breathing exercises to get rid of the inflammation liquid. The idea is to come with an app to perform breathing exercises to the user. The app is going to check if the user is doing the breathing exercises right. For the app development we are planning to use reackt native, for computer vision part we are still looking for the best option because it needs to be simple yet able to detect the expension of the chest."

  @ViewChild('record') video: ElementRef;
  @ViewChild('container') container: ElementRef;

  public constructor(private videoService: VideoService) {
  }

  ngOnInit() { }

  ngAfterViewInit() {
    // if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    //   navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    //     // this.video.nativeElement.srcObject = stream;
    //     // this.video.nativeElement.play();
    //     // this.postStream(this.video.nativeElement.srcObject);
    //     this.cameraAllowed = true;
    //   });
    // }
  }

  postStream(stream) {

    this.videoService.send(stream).subscribe(
        data => {
          // this.currentGuideline = data;
        },
        err => {
          console.log(err)
          if (err.status > 300) {
          }
        },
        () => {
          console.log('Request Completed');
          // this.addGuidelineMode = false
          // this.addQuestionMode = true;
          // this.currentAnswer = {}
        }
    );
  }

  onDisplayClick() {
    this.displayCamera = true;
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        // this.video.nativeElement.srcObject = stream;
        // this.video.nativeElement.play();
        // this.postStream(this.video.nativeElement.srcObject);
        this.cameraAllowed = true;
      });
    }
    // this.scrollTo();
  }

  scrollTo() {
    window.scrollTo(0, this.container.nativeElement.scrollHeight);
    // this.image.nativeElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

}
