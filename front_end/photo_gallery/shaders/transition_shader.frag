precision mediump float;

uniform sampler2D uTexture;
uniform float uProgress;

void main() {
  vec2 uv = gl_FragCoord.xy / resolution.xy;
  vec4 color = texture2D(uTexture, uv);
  
  // Example effect: simple wave distortion
  float wave = sin(uv.y * 10.0 + uProgress * 3.14) * 0.1;
  uv.x += wave * uProgress;
  
  gl_FragColor = texture2D(uTexture, uv);
} 