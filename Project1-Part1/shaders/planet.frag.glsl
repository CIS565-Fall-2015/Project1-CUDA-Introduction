#version 330

out vec4 fragColor;

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main() {
    fragColor = vec4( rand(gl_FragCoord.yz) ,rand(gl_FragCoord.zx),rand(gl_FragCoord.xy),1.0);
}