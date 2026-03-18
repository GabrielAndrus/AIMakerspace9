/* ══════════════════════════════════════════════════════════════
   SLIDE 1: TITLE — Typewriter
   ══════════════════════════════════════════════════════════════ */

let slide1Entered = false;
registerAnim(0,
  function enter() {
    if (!slide1Entered) {
      slide1Entered = true;
      setTimeout(() => {
        typewriter(
          document.getElementById('typewriter-target'),
          'You have a spreadsheet of Titanic passengers. You want to predict who survived. <strong style="color:#b8bb26;">You shouldn\'t need a PhD for that.</strong>',
          40
        );
      }, 400);
    }
  },
  null
);
