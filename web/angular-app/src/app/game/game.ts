import { Component } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-game',
  standalone: false,
  templateUrl: './game.html',
  styleUrl: './game.scss',
})
export class Game {
  // Le build Pygbag est servi depuis public/game/index.html
  gameUrl: SafeResourceUrl;

  constructor(private sanitizer: DomSanitizer) {
    this.gameUrl = this.sanitizer.bypassSecurityTrustResourceUrl('/game/index.html');
  }
}
