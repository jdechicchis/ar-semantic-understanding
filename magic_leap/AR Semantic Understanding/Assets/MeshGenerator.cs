using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MeshGenerator : MonoBehaviour
{
    private Mesh mesh;

    private Vector3[] vertices;
    private int[] triangles;

    // Start is called before the first frame update
    void Start()
    {
        this.mesh = new Mesh();
        this.GetComponent<MeshFilter>().mesh = this.mesh;

        CreateMesh();
        UpdateMesh();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void CreateMesh()
    {
        this.vertices = new Vector3[]
        {
            new Vector3(0, 0, 0),
            new Vector3(0, 0, 1),
            new Vector3(1, 0, 0)
        };

        this.triangles = new int[]
        {
            0, 1, 2
        };
    }

    void UpdateMesh ()
    {
        this.mesh.Clear();

        this.mesh.vertices = this.vertices;
        this.mesh.triangles = triangles;
    }
}
