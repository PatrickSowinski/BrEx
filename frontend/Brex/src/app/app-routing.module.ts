import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

import { HomeComponent } from './home/home.component';
import { ExerciseComponent } from './exercise/exercise.component';
import { SolutionComponent } from './solution/solution.component';
import { PressComponent } from './press/press.component';


const routes: Routes = [
  { path: 'home', component: HomeComponent },
  { path: 'exercise', component: ExerciseComponent },
  { path: 'solution', component: SolutionComponent },
  { path: 'press', component: PressComponent },
  { path: '', redirectTo: '/home', pathMatch: 'full' },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
