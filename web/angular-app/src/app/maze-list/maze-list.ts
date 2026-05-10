import { Component, OnInit, Output, EventEmitter } from '@angular/core';
import { ApiService, MazeSummary } from '../services/api';

@Component({
  selector: 'app-maze-list',
  standalone: false,
  templateUrl: './maze-list.html',
  styleUrl: './maze-list.scss',
})
export class MazeList implements OnInit {
  /** Émet le seed sélectionné vers le parent (AppComponent) */
  @Output() seedSelected = new EventEmitter<number>();

  mazes: MazeSummary[] = [];
  loading = false;
  error: string | null = null;

  // Filtres
  limit = 20;
  ratedOnly = false;
  minScore = 0;

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.load();
  }

  load(): void {
    this.loading = true;
    this.error = null;
    this.api.getMazes(this.limit, this.ratedOnly, this.minScore).subscribe({
      next: (mazes) => {
        this.mazes = mazes;
        this.loading = false;
      },
      error: (err) => {
        this.error = `Erreur : ${err.message ?? 'impossible de contacter l\'API'}`;
        this.loading = false;
      },
    });
  }

  select(seed: number): void {
    this.seedSelected.emit(seed);
  }

  stars(rating: number | null): string {
    if (rating === null) return '—';
    return '★'.repeat(rating) + '☆'.repeat(5 - rating);
  }
}
