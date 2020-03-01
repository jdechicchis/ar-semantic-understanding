Shader "Custom/PointCloudShader"
{
    Properties     
    {
        _PointSize("Point size", Float) = 5.0
    }
    SubShader
    {         
        Pass 
        { 
            LOD 200

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            uniform float _PointSize;

            struct VertexInput
            {
                float4 vertex: POSITION;
                float4 color: COLOR;
            };

            struct VertexOutput          
            {
                float4 pos: SV_POSITION;     
                float4 col: COLOR;
                float size: PSIZE;
            };

            VertexOutput vert(VertexInput v)
            {
                VertexOutput o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.col = v.color;
                o.size = _PointSize;

                return o;             
            }

            float4 frag(VertexOutput o) : COLOR
            {
                return o.col;             
            }
            ENDCG
          }     
    }
}
