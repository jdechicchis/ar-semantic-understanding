using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Control : MonoBehaviour
{
    public GameObject WorldCanvas;

    private HeadLockScript headlock;
    // Start is called before the first frame update
    void Start()
    {
        this.headlock = GetComponentInChildren<HeadLockScript>();
    }

    // Update is called once per frame
    void Update()
    {
        this.headlock.HardHeadLock(this.WorldCanvas);
    }
}
