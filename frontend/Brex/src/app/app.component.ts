import { Component, OnInit, AfterViewInit } from '@angular/core';

import { VideoService } from './services/video.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit, AfterViewInit {
  title = 'Brex';
  baseUrl = 'http://localhost:5000/video_feed';
  imgSrc;

  constructor(private videoService: VideoService) {}

  ngOnInit(): void {
    // this.getVideoStream();
  }

  ngAfterViewInit() {
    // this.getVideoStream();
  }

  getVideoStream() {
    this.videoService.getVideo().subscribe(
      data => {
        this.imgSrc = data;
      });
  }
}
