in vec3 v_Color;
uniform vec4 u_Tint;
out vec4 FragColor;

void main()
{
    FragColor = vec4(v_Color * u_Tint.rgb, u_Tint.a);
}
