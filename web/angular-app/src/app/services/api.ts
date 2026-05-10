import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface MazeSummary {
  seed: number;
  complexity_score: number | null;
  rating: number | null;
  created_at: string | null;
}

export interface RatingResponse {
  seed: number;
  rating: number;
  message: string;
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  // En dev, le proxy Angular redirige /api → https://pacman-s2-h.onrender.com
  private readonly base = '/api';

  constructor(private http: HttpClient) {}

  getMazes(limit = 20, ratedOnly = false, minScore = 0): Observable<MazeSummary[]> {
    const params = new HttpParams()
      .set('limit', limit)
      .set('rated_only', ratedOnly)
      .set('min_score', minScore);
    return this.http.get<MazeSummary[]>(`${this.base}/mazes`, { params });
  }

  rateMaze(seed: number, rating: number): Observable<RatingResponse> {
    return this.http.patch<RatingResponse>(
      `${this.base}/maze/${seed}/rate`, { rating }
    );
  }
}
