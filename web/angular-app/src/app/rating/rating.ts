import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { ApiService } from '../services/api';

@Component({
  selector: 'app-rating',
  standalone: false,
  templateUrl: './rating.html',
  styleUrl: './rating.scss',
})
export class Rating implements OnChanges {
  /** Seed du maze à noter (transmis par le parent via [seed]="…") */
  @Input() seed: number | null = null;

  selectedRating: number | null = null;
  confirmed = false;
  sending = false;
  message: string | null = null;
  error: string | null = null;

  readonly ratingLabels = ['Nul', 'Mauvais', 'Moyen', 'Bien', 'Très bien', 'Excellent'];
  readonly ratingValues = [0, 1, 2, 3, 4, 5];

  constructor(private api: ApiService) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['seed']) {
      this.reset();
    }
  }

  select(value: number): void {
    if (!this.confirmed) {
      this.selectedRating = value;
    }
  }

  submit(): void {
    if (this.seed === null || this.selectedRating === null || this.sending) return;
    this.sending = true;
    this.error = null;
    this.api.rateMaze(this.seed, this.selectedRating).subscribe({
      next: (res) => {
        this.confirmed = true;
        this.sending = false;
        this.message = `Note ${res.rating}/5 enregistrée pour le seed ${res.seed}.`;
      },
      error: (err) => {
        this.sending = false;
        this.error = err.error?.error ?? 'Erreur lors de l\'envoi.';
      },
    });
  }

  reset(): void {
    this.selectedRating = null;
    this.confirmed = false;
    this.sending = false;
    this.message = null;
    this.error = null;
  }
}
