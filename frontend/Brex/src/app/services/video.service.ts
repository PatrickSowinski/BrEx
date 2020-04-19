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
    return this.http.get(this.baseUrl + 'video_feed');
  }

  public send(data: any): any {
    return this.http.post(this.baseUrl + 'stream_video', data).pipe(map(response => response));
  }

  public getStart(): any {
    return this.http.get(this.baseUrl + 'success').pipe(map(response => response));
  }

  public getCalculations(): any {
    return this.http.get(this.baseUrl + 'calculations').pipe(map(response => response));
  }

  public getPlot(): any {
    return this.http.get(this.baseUrl + 'plot');
  }
}
