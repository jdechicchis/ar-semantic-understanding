using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum Mode { LOOSE, SOFT, HARD }

public class Control : MonoBehaviour
{
    public Mode WorldMode;
    public GameObject WorldCanvas;
    public GameObject Light;
    public GameObject Camera;

    private HeadLockScript headlock;

    // Start is called before the first frame update
    void Start()
    {
        this.headlock = GetComponentInChildren<HeadLockScript>();
    }

    // Update is called once per frame
    void Update()
    {
        this.headLockAndLight();
    }

    private void headLockAndLight()
    {
        if (this.WorldMode == Mode.SOFT)
        {
            this.headlock.HeadLock(this.WorldCanvas, 5.0f);
        }
        else if (this.WorldMode == Mode.LOOSE)
        {
            this.headlock.HeadLock(this.WorldCanvas, 1.75f);
        }
        else if (this.WorldMode == Mode.HARD)
        {
            this.headlock.HardHeadLock(this.WorldCanvas);
        }
        else
        {
            throw new Exception("Invalid world mode!");
        }
    }
}