import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';

import {WebcamModule} from 'ngx-webcam';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import {VideoService} from './services/video.service';
import { HomeComponent } from './home/home.component';
import { ExerciseComponent } from './exercise/exercise.component';
import { SolutionComponent } from './solution/solution.component';
import { PressComponent } from './press/press.component';

@NgModule({
  declarations: [
    AppComponent,
    HomeComponent,
    ExerciseComponent,
    SolutionComponent,
    PressComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    WebcamModule,
    HttpClientModule
  ],
  providers: [VideoService],
  bootstrap: [AppComponent]
})
export class AppModule { }
