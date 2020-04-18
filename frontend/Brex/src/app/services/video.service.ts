import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Inject } from '@angular/core';
import { map } from 'rxjs/operators';

@Injectable({
  providedIn: 'root',
})
export class VideoService {
  public baseUrl = 'http://localhost:5000/'

  constructor(
    @Inject(HttpClient) protected http: HttpClient,
  ) {
  }

  public getVideo(): any {
    return this.http.get('http://localhost:5000/video_feed');
  }
}
