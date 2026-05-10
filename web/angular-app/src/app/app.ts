import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.html',
  standalone: false,
  styleUrl: './app.scss',
})
export class App {
  selectedSeed: number | null = null;

  onSeedSelected(seed: number): void {
    this.selectedSeed = seed;
  }
}
